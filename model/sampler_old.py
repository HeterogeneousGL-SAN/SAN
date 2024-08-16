import os.path
from args_list import get_args
from utils import create_graph_csv
import random
import networkx as nx
import walker
import multiprocessing as mp
from collections import Counter
import time
import pickle
import numpy as np
from loader import ScholarlyDataset
from sklearn.preprocessing import normalize
import pandas as pd
import torch
import numpy as np
import torch
import random
import multiprocessing
import concurrent.futures

from multiprocessing import Pool
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.enabled = False


def cosine_similarity(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_vector1 = np.linalg.norm(emb1)
    norm_vector2 = np.linalg.norm(emb2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


class RandomWalkWithRestart:
    """
    args:
    - dataset
    - setup -- inductive light (il), inductive full (if), inductive hard (ih), transductive (t)
    - set -- train, validation, test
    - max number of paths per node -- the paths selected to choose neighbours from -- default 10
    - number of selected neighbours
    - number of core nodes
    - max path's length
    - seed_node
    - restart prob
    - nx_graph
    - distance penalty -- the penalty applied to distant nodes
    - reward frequency -- the reward applied to frequent nodes

    """

    def __init__(self, args, sch_dataset, root, set_type=None):
        # root = sch_dataset.root
        # sch_dataset = sch_dataset[0]
        self.args = args
        self.set_type = set_type
        self.path_data = f'datasets/{args.dataset}/all/final'
        self.data = sch_dataset
        self.max_walk_length = args.max_walk_length
        self.all_walks = args.num_random_walks  # at the beginning compute 100 walks per node and store them, sampling will be random on these walks
        self.alpha = args.restart_probability

        self.G = create_graph_csv(self.path_data, self.args.dataset, ['all'])
        # print(f'total detected nodes {len(self.G.nodes())}')
        # print(f'total detected edges {len(self.G.edges())}')

        # Filter edges based on the selected source and target nodes
        if 'transductive' in root:
            # rimuovo gli archi tra p e d, li aggiungo dopo dipendentemente da cosa mi serve
            # qua sono in transduttivo e divido gli archi per il message passing, tenendo solo la kcore per il mp
            if 'train' in root:
                edge_index = sch_dataset['publication', 'cites', 'dataset'].edge_index_train
            elif 'validation' in root:
                edge_index = sch_dataset['publication', 'cites', 'dataset'].edge_index_validation
            elif 'test' in root:
                edge_index = sch_dataset['publication', 'cites', 'dataset'].edge_index_test
            else:
                exit(1)
            edges_to_remove = [(u, v) for u, v in self.G.edges if
                               (u.startswith('p_') and v.startswith('d_')) or (
                                           u.startswith('d_') and v.startswith('p'))]

            # rimuovo tutti gli edges tra p e d e aggiungo quelli che ho trovato sopra
            edges_to_add = [
                tuple([sch_dataset['publication'].rev_mapping[s[0]], sch_dataset['dataset'].rev_mapping[s[1]]]) for s in
                edge_index.t().tolist()]
            # print(edges_to_add[0:10])
            # print(f'edges to remove {len(edges_to_remove)}')
            # print(f'edges to keep {len(edges_to_add)}')
            self.G.remove_edges_from(edges_to_remove)
            # print(f'total detected nodes {len(self.G.nodes())}')
            # print(f'total detected edges {len(self.G.edges())}')
            self.G.add_edges_from(edges_to_add)
            # print(f'total detected nodes {len(self.G.nodes())}')
            # print(f'total detected edges {len(self.G.edges())}')

        # self.seeds = list(sch_dataset['publication'].rev_mapping.values()) + list(sch_dataset['dataset'].rev_mapping.values())
        # self.seeds = self.seeds[0:300]
        # self.seeds = ['p_0','p_1','d_3010']
        if args.inductive:
            if set_type == 'train':
                print('sono in train')
                # in questo caso rimuovo tutti i nodi che appaiono in test e validation (p e d)
                edge_index_vali = sch_dataset['publication', 'cites', 'dataset'].edge_label_index_validation
                edge_index_test = sch_dataset['publication', 'cites', 'dataset'].edge_label_index_test
                sources = list(set(edge_index_vali[0].tolist() + edge_index_test[0].tolist()))
                targets = list(set(edge_index_vali[1].tolist() + edge_index_test[1].tolist()))
                sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
                targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
                self.G.remove_nodes_from(sources)
                if args.inductive_type == 'full':
                    self.G.remove_nodes_from(targets)
            elif set_type == 'validation':
                # in questo caso rimuovo tutti i nodi che appaiono in test e validation (p e d)
                edge_index_test = sch_dataset['publication', 'cites', 'dataset'].edge_label_index_test
                sources = list(set(edge_index_test[0].tolist()))
                targets = list(set(edge_index_test[1].tolist()))
                sources = [sch_dataset['publication'].rev_mapping[s] for s in sources]
                targets = [sch_dataset['dataset'].rev_mapping[s] for s in targets]
                self.G.remove_nodes_from(sources)
                if args.inductive_type == 'full':
                    self.G.remove_nodes_from(targets)

        self.G_conv = nx.convert_node_labels_to_integers(self.G)
        self.mapping = {j: i for j, i in zip(self.G_conv, self.G)}
        self.rev_mapping = {i: j for j, i in zip(self.G_conv, self.G)}

        self.publications_features, self.datasets_features = sch_dataset['publication'], sch_dataset['dataset']
        # self.publications_features,self.datasets_features = self.load_features()

    def get_seeds(self):
        return self.seeds

    def reset_seeds(self):
        self.seeds = False

    def create_random_walks(self, seeds_in=False, all=False):
        if all:
            seeds = [node for node in self.G.nodes() if self.is_publication(node) or self.is_dataset(node)]
            self.seeds = seeds
        else:
            if not seeds_in:
                seeds = self.seeds
            else:
                self.seeds = seeds_in
                seeds = seeds_in

        if seeds is not None:
            converted_seeds = [self.rev_mapping[x] for x in self.seeds]
            # print(converted_seeds)
            walks = walker.random_walks(self.G_conv, n_walks=self.all_walks, alpha=0.1,
                                        walk_len=self.args.max_walk_length, start_nodes=converted_seeds, q=0.1, p=0.3)
        else:
            walks = walker.random_walks(self.G_conv, n_walks=self.all_walks, alpha=self.alpha,
                                        walk_len=random.randint(1, self.max_walk_length), p=1.0, q=0.0)
        walks = sorted(walks, key=lambda x: x[0])
        walks_rev = []
        for walk in walks:
            walks_rev.append([self.mapping[p] for p in walk])

        # print(f'Walks generated in: {time.time() - st}')
        # print('Saving walks')
        # file = open(f'model/data/{args.dataset}/random_walks/random_walks_{epoch}.txt', 'w')
        # for walk in walks_rev:
        #     print(walk)
        #     file.write(' '.join(walk) + '\n')

        return walks_rev

    def set_seeds(self, seeds=None):
        if seeds is None:
            seeds = [node for node in self.G.nodes() if self.is_publication(node) or self.is_dataset(node)]
            self.seeds = seeds
        else:
            self.seeds = seeds

    def is_publication(self, node):
        return node.startswith('p_')

    def is_dataset(self, node):
        return node.startswith('d_')

    def is_entity(self, node):
        return node.startswith('dbpedia_')

    def is_keyword(self, node):
        return node.startswith('k_')

    def is_topic(self, node):
        return node.startswith('t')

    def is_author(self, node):
        return node.startswith('a_')

    def is_venue(self, node):
        return node.startswith('v_')

    def is_organization(self, node):
        return node.startswith('o_')

    def get_walks_threshold(self, walk_scores):
        return np.percentile(walk_scores, 75)

    def process_list(self,seeds_list):
        # Example function to process each sublist
        selected_walks_list, selected_cores_list, selected_hubs_top_list, selected_hubs_key_list = [],[],[],[]
        print(f"Processing sublist of length {len(seeds_list)}")

        # Replace this with the actual work you need to do on each sublist
        walk_importance_score = [[] for _ in range(len(seeds_list))]
        cores = [[] for _ in range(len(seeds_list))]
        hubs_top = [[] for _ in range(len(seeds_list))]
        hubs_key = [[] for _ in range(len(seeds_list))]

        for j,seed in enumerate(seeds_list):
            if self.is_publication(seed):
                seed_id = self.publications_features['mapping'][seed]
                seed_id = self.publications_features.x[seed_id]
            else:
                seed_id = self.datasets_features['mapping'][seed]
                seed_id = self.datasets_features.x[seed_id]

            ind_seed = self.first_ind.index(seed)
            cur_walks = self.walks[ind_seed: (ind_seed) + self.all_walks]
            for i, w in enumerate(cur_walks):
                walk = []
                for node in w:
                    if node not in walk:
                        walk.append(node)
                cores_walk = []
                hubs_key_walk = []
                hubs_top_walk = []
                neighs = walk[1:]
                cosine_total = 0
                neigh_count = 0
                for n in neighs:
                    if n == seed:
                        neigh_count = 1
                    else:
                        neigh_count += 1
                        frequency = len([a for a in walk if a == n])
                        distance = neigh_count
                        penalty = frequency / distance
                        if self.is_publication(n) or self.is_dataset(n):
                            if self.is_publication(n):
                                n_id = self.publications_features.mapping[n]
                                n_id = self.publications_features.x[n_id]
                            else:
                                n_id = self.datasets_features.mapping[n]
                                n_id = self.datasets_features.x[n_id]

                            cosine_0 = cosine_similarity(seed_id, n_id)
                            cosine = cosine_0 * penalty
                            cores_walk.append((n, cosine_0))

                            cosine_total += cosine
                        elif self.is_author(n) or self.is_organization(n) or self.is_venue(n):
                            hubs_top_walk.append((n, penalty))
                        else:
                            hubs_key_walk.append((n, penalty))

                if len(cores_walk) == 0:
                    cosine_total = 0
                else:
                    cosine_total = cosine_total / len(cores_walk)
                walk_importance_score[j].append(cosine_total)
                cores[j].append(cores_walk)
                hubs_top[j].append(hubs_top_walk)
                hubs_key[j].append(hubs_key_walk)

            for index in range(self.args.num_selected_walks, self.args.num_random_walks):
                selected_walks_ind = sorted(range(len(walk_importance_score[j])),
                                            key=lambda i: walk_importance_score[j][i], reverse=True)[:index]
                selected_walks = [cur_walks[i] for i in selected_walks_ind]
                selected_cores = [cores[j][i] for i in selected_walks_ind]
                selected_cores = list(set(item for sublist in selected_cores for item in sublist))

                selected_hubs_top = [hubs_top[j][i] for i in selected_walks_ind]
                selected_hubs_top = list(set(item for sublist in selected_hubs_top for item in sublist))

                selected_hubs_key = [hubs_key[j][i] for i in selected_walks_ind]
                selected_hubs_key = list(set(item for sublist in selected_hubs_key for item in sublist))

                if (len(selected_cores) >= self.args.n_cores and
                        len(selected_hubs_key) >= self.args.n_keys_hubs and
                        len(selected_hubs_top) >= self.args.n_top_hubs):
                    break
            selected_walks_list.append(selected_walks)
            selected_cores_list.append(selected_cores)
            selected_hubs_top_list.append(selected_hubs_top)
            selected_hubs_key_list.append(selected_hubs_key)

        # print('\n\n')
        # print(len(selected_walks_list))
        # print(len(selected_cores_list))
        # print(len(selected_hubs_top_list))
        # print(len(selected_hubs_key_list))
        # print('\n\n')
        # print(selected_walks_list[0])
        # print(selected_cores_list[0])
        # print('\n\n')

        return [selected_walks_list, selected_cores_list, selected_hubs_top_list, selected_hubs_key_list]



    def select_walks_mp(self, walks):

        """Select the walks such that
            - the mean cos sim is maximized
            - min num neighbours is reached
        """

        def chunk_list(main_list, chunk_size):
            # Divide the main list into chunks of specified size
            for i in range(0, len(main_list), chunk_size):
                yield main_list[i:i + chunk_size]

        # cores_scores contains the cos. sim. for each p and d in each walk
        # print('walks selection started')
        hubs_key = [[[] for _ in range(self.all_walks)] for _ in range(len(self.seeds))]
        hubs_top = [[[] for _ in range(self.all_walks)] for _ in range(len(self.seeds))]
        cores = [[[] for _ in range(self.all_walks)] for _ in range(len(self.seeds))]
        selected_seeds_cores = [[] for _ in range(len(self.seeds))]
        selected_seeds_hubs_top = [[] for _ in range(len(self.seeds))]
        selected_seeds_hubs_key = [[] for _ in range(len(self.seeds))]

        selected_seeds_walks = [[] for _ in range(len(self.seeds))]
        walk_importance_score = [[] for _ in range(len(self.seeds))]  # keeps count of node distance!
        all_nodes = []
        # compute frequency
        for i, walk in enumerate(walks):
            all_nodes.extend(walk)
        counts = Counter(all_nodes)
        frequency_dict = dict(counts)
        # group walks by seed
        # print(f'total number of walks: {len(walks)}')
        st = time.time()
        first_ind = [w[0] for w in walks]
        self.first_ind = first_ind
        self.walks = walks

        # Number of elements in each chunk
        chunk_size = 2000

        # Divide the main list into 35 sublists of 1000 elements each
        sublists = list(chunk_list(self.seeds, chunk_size))
        # print(len(sublists))
        # Number of processes (ideally should be equal to or less than the number of available CPU cores)
        num_processes = min(40, len(sublists))
        st = time.time()
        # Use ProcessPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Map the process_list function to each sublist
            results = list(executor.map(self.process_list, sublists))



        # st =time.time()
        # args = [(j, seed,first_ind, walks, self.args, self.seeds, self.publications_features, self.datasets_features,
        #          self.is_publication, self.is_author, self.is_organization, self.is_venue,
        #           self.args.n_cores, self.args.n_keys_hubs, self.args.n_top_hubs,
        #          self.args.num_selected_walks, self.args.num_random_walks) for j, seed in enumerate(self.seeds)]
        # results = pool.map(self.process_seed, args)
        # pool.close()
        # pool.join()
        end =time.time()
        # print(f'time taken by multi: {end-st}')
        res_concat = []
        j = 0
        # sono i risultati di ogni chunk
        for result in results:
            # qui è ogni chunk. Ogni chunk è formato da walks, cores, top, keys


            walk_list, core_list, hubtop_list, hubkey_list = result[0],result[1],result[2],result[3]
            for walk,core,hubtop,hubkey in zip(walk_list,core_list,hubtop_list,hubkey_list):

                selected_seeds_walks[j] = walk
                selected_seeds_cores[j] = core
                selected_seeds_hubs_top[j] = hubtop
                selected_seeds_hubs_key[j] = hubkey
                if walk[0][0] == 'p_1111111110':
                    print('IN')
                    print(len(result))
                    print(len(walk))
                    print(walk[0])
                    print(walk)
                    print(len(core))
                    print(core)
                    print(core[0])
                    print(len(hubtop))
                    print(hubtop[0])
                    print(len(hubkey))
                    print(hubkey[0])
                j += 1

        # print(len(selected_seeds_walks))
        # print(len(self.seeds))
        # print(f'walks: {selected_seeds_walks[0]}')
        # print(f'cores: {selected_seeds_cores[0]}')
        final_cores, final_hubs_key, final_hubs_top = [], [], []
        # now, I select the top-k elements basing on the similarity score with penalty
        selected_seeds_cores = [sorted(s, key=lambda x: x[1], reverse=True) for s in selected_seeds_cores]
        # print(selected_seeds_cores)
        for i, s in enumerate(selected_seeds_cores):

            final_cores.append(
                sorted(list({x[0]: x for x in reversed(s)}.values())[::-1], key=lambda p: (p[1], p[0]), reverse=True))

        selected_seeds_hubs_top = [sorted(s, key=lambda x: x[1], reverse=True) for s in selected_seeds_hubs_top]
        for i, s in enumerate(selected_seeds_hubs_top):
            final_hubs_top.append(
                sorted(list({x[0]: x for x in reversed(s)}.values())[::-1], key=lambda p: (p[1], p[0]), reverse=True)[
                :self.args.n_top_hubs])

        selected_seeds_hubs_key = [sorted(s, key=lambda x: x[1], reverse=True) for s in selected_seeds_hubs_key]
        for i, s in enumerate(selected_seeds_hubs_key):

            final_hubs_key.append(
                sorted(list({x[0]: x for x in reversed(s)}.values())[::-1], key=lambda p: (p[1], p[0]), reverse=True)[
                :self.args.n_keys_hubs])

        return selected_seeds_walks, final_cores, final_hubs_key, final_hubs_top






    def select_walks(self, walks):

        """Select the walks such that
            - the mean cos sim is maximized
            - min num neighbours is reached
        """
        # cores_scores contains the cos. sim. for each p and d in each walk
        # print('walks selection started')
        hubs_key = [[[] for _ in range(self.all_walks)] for _ in range(len(self.seeds))]
        hubs_top = [[[] for _ in range(self.all_walks)] for _ in range(len(self.seeds))]
        cores = [[[] for _ in range(self.all_walks)] for _ in range(len(self.seeds))]
        selected_seeds_cores = [[] for _ in range(len(self.seeds))]
        selected_seeds_hubs_top = [[] for _ in range(len(self.seeds))]
        selected_seeds_hubs_key = [[] for _ in range(len(self.seeds))]

        selected_seeds_walks = [[] for _ in range(len(self.seeds))]
        walk_importance_score = [[] for _ in range(len(self.seeds))]  # keeps count of node distance!
        all_nodes = []
        # compute frequency
        for i, walk in enumerate(walks):
            all_nodes.extend(walk)
        counts = Counter(all_nodes)
        frequency_dict = dict(counts)
        # group walks by seed
        # print(f'total number of walks: {len(walks)}')
        st = time.time()
        first_ind = [w[0] for w in walks]
        st = time.time()
        # print('SEEDS: ',len(self.seeds))
        for j, seed in enumerate(self.seeds):
            if self.is_publication(seed):
                seed_id = self.publications_features['mapping'][seed]
                seed_id = self.publications_features.x[seed_id]
            else:
                seed_id = self.datasets_features['mapping'][seed]
                seed_id = self.datasets_features.x[seed_id]


            # cur_walks = [walk for walk in walks if walk[0] == seed]
            # cur_walks = sorted(walks, key=lambda x: x[0])
            ind_seed = first_ind.index(seed)
            cur_walks = walks[ind_seed : (ind_seed)+self.args.num_random_walks]
            for i, w in enumerate(cur_walks):

                # walk = list(set(w))
                walk = []
                for node in w:
                    if node not in walk:
                        walk.append(node)

                cores_walk = []
                hubs_key_walk = []
                hubs_top_walk = []
                neighs = walk[1::]
                cosine_total = 0
                neigh_count = 0
                for n in neighs:
                    if n == seed:
                        neigh_count = 1
                    else:
                        neigh_count += 1
                        # frequency = frequency_dict[n]
                        frequency = len([a for a in walk if a == n])
                        distance = neigh_count
                        penalty = frequency / distance
                        if self.is_publication(n) or self.is_dataset(n):
                            if self.is_publication(n):
                                n_id = self.publications_features.mapping[n]
                                n_id = self.publications_features.x[n_id]
                            else:
                                n_id = self.datasets_features.mapping[n]
                                n_id = self.datasets_features.x[n_id]

                            cosine_0 = cosine_similarity(seed_id, n_id)
                            cosine = cosine_0 * penalty
                            cores_walk.append(tuple([n, cosine_0]))

                            if self.args.verbose_sampler:
                                print(n, frequency_dict[n], distance, penalty, cosine_0, cosine)
                            cosine_total += cosine
                        elif self.is_author(n) or self.is_organization(n) or self.is_venue(n):
                            hubs_top_walk.append(tuple([n, penalty]))
                        else:
                            hubs_key_walk.append(tuple([n, penalty]))
                        # distance_walk.append(tuple([n, distance]))
                        # frequency_walk.append(tuple([n, frequency]))
                if len(cores_walk) == 0:
                    cosine_total = 0
                else:
                    cosine_total = cosine_total / len(cores_walk)
                walk_importance_score[j].append(cosine_total)
                # print(f'total cosine: {cosine_total}\n')
                cores[j][i] = cores_walk
                hubs_top[j][i] = hubs_top_walk
                # distances[j][i] = distance_walk
                # frequencies[j][i] = frequency_walk
                hubs_key[j][i] = hubs_key_walk
            # this for cycle prevents having too few nodes
            for index in range(self.args.num_selected_walks, self.args.num_random_walks):
                selected_walks_ind = sorted(range(len(walk_importance_score[j])),
                                            key=lambda i: walk_importance_score[j][i], reverse=True)[0:index]

                selected_walks = [cur_walks[i] for i in selected_walks_ind]
                selected_cores = [cores[j][i] for i in selected_walks_ind]
                selected_cores = list(set([item for sublist in selected_cores for item in sublist]))

                selected_hubs_top = [hubs_top[j][i] for i in selected_walks_ind]
                selected_hubs_top = list(set([item for sublist in selected_hubs_top for item in sublist]))

                selected_hubs_key = [hubs_key[j][i] for i in selected_walks_ind]
                selected_hubs_key = list(set([item for sublist in selected_hubs_key for item in sublist]))
                if len(selected_cores) >= self.args.n_cores and len(selected_hubs_key) >= self.args.n_keys_hubs and len(
                        selected_hubs_top) >= self.args.n_top_hubs:
                    break

            selected_seeds_walks[j] = selected_walks
            selected_seeds_cores[j] = selected_cores
            selected_seeds_hubs_top[j] = selected_hubs_top
            selected_seeds_hubs_key[j] = selected_hubs_key
            # if seed == 'p_0':
            #     print('IN')
            #     print(len(selected_walks))
            #     print(selected_walks)
            #     print(len(selected_cores))
            #     print(selected_cores)

            # if seed == 'd_1004':
            #     print(selected_walks)
            #     print(selected_walks)
            #     print(selected_hubs_key)
            #     print(sorted(list({x[0]: x for x in reversed(selected_hubs_key)}.values())[::-1], key=lambda p: (p[1], p[0]), reverse=True))
            #     print('\n\n')
        end = time.time()
        print(f'time taken: {end-st}')

        final_cores, final_hubs_key, final_hubs_top = [], [], []
        # now, I select the top-k elements basing on the similarity score with penalty
        selected_seeds_cores = [sorted(s, key=lambda x: x[1], reverse=True) for s in selected_seeds_cores]
        # print(selected_seeds_cores)
        for i, s in enumerate(selected_seeds_cores):

            final_cores.append(
                sorted(list({x[0]: x for x in reversed(s)}.values())[::-1], key=lambda p: (p[1], p[0]), reverse=True))

        selected_seeds_hubs_top = [sorted(s, key=lambda x: x[1], reverse=True) for s in selected_seeds_hubs_top]
        for i, s in enumerate(selected_seeds_hubs_top):
            final_hubs_top.append(
                sorted(list({x[0]: x for x in reversed(s)}.values())[::-1], key=lambda p: (p[1], p[0]), reverse=True)[
                :self.args.n_top_hubs])

        selected_seeds_hubs_key = [sorted(s, key=lambda x: x[1], reverse=True) for s in selected_seeds_hubs_key]
        for i, s in enumerate(selected_seeds_hubs_key):

            final_hubs_key.append(
                sorted(list({x[0]: x for x in reversed(s)}.values())[::-1], key=lambda p: (p[1], p[0]), reverse=True)[
                :self.args.n_keys_hubs])

        return selected_seeds_walks, final_cores, final_hubs_key, final_hubs_top

    def get_neighbours_vector(self, sel_cores, sel_keys, sel_top):

        publications_vectors = [[] for _ in range(len(self.seeds))]
        publications_net_vectors = [[] for _ in range(len(self.seeds))]
        datasets_vectors = [[] for _ in range(len(self.seeds))]
        datasets_net_vectors = [[] for _ in range(len(self.seeds))]
        authors_vectors = [[] for _ in range(len(self.seeds))]
        venues_vectors = [[] for _ in range(len(self.seeds))]
        keywords_vectors = [[] for _ in range(len(self.seeds))]
        organizations_vectors = [[] for _ in range(len(self.seeds))]
        entities_vectors = [[] for _ in range(len(self.seeds))]
        topics_vectors = [[] for _ in range(len(self.seeds))]

        ret_cores = [[] for _ in range(len(self.seeds))]
        ret_net_cores = [[] for _ in range(len(self.seeds))]
        ret_tops = [[] for _ in range(len(self.seeds))]
        ret_keys = [[] for _ in range(len(self.seeds))]
        ret_net_keys = [[] for _ in range(len(self.seeds))]
        vectors = []
        vectors_net = []
        for i, seed in enumerate(self.seeds):
            # print(seed)
            if self.is_publication(seed):
                vectors.append([seed, seed, 1.0, self.data['publication'].mapping[seed],
                                self.data['publication'].x[self.data['publication'].mapping[seed]]])
                vectors_net.append([seed, seed, 1.0, self.data['publication'].mapping[seed],
                                    self.data['publication'].net_x[self.data['publication'].mapping[seed]]])
            elif self.is_dataset(seed):
                vectors.append([seed, seed, 1.0, self.data['dataset'].mapping[seed],
                                self.data['dataset'].x[self.data['dataset'].mapping[seed]]])
                vectors_net.append([seed, seed, 1.0, self.data['dataset'].mapping[seed],
                                    self.data['dataset'].net_x[self.data['dataset'].mapping[seed]]])

            cores = sel_cores[i]
            publications_net = [[seed, c[0], c[1], self.data['publication'].mapping[c[0]],
                                 self.data['publication'].net_x[self.data['publication'].mapping[c[0]]]] for c in cores
                                if self.is_publication(c[0])]
            publications = [[seed, c[0], c[1], self.data['publication'].mapping[c[0]],
                             self.data['publication'].x[self.data['publication'].mapping[c[0]]]] for c in cores if
                            self.is_publication(c[0])]
            datasets_net = [[seed, c[0], c[1], self.data['dataset'].mapping[c[0]],
                             self.data['dataset'].net_x[self.data['dataset'].mapping[c[0]]]] for c in cores if
                            self.is_dataset(c[0])]

            datasets = [[seed, c[0], c[1], self.data['dataset'].mapping[c[0]],
                         self.data['dataset'].x[self.data['dataset'].mapping[c[0]]]] for c in cores if
                        self.is_dataset(c[0])]

            tops = sel_top[i]
            authors = [[seed, c[0], c[1], self.data['author'].mapping[c[0]],
                        self.data['author'].net_x[self.data['author'].mapping[c[0]]]] for c in tops if
                       self.is_author(c[0])]
            venues = [[seed, c[0], c[1], self.data['venue'].mapping[c[0]],
                       self.data['venue'].net_x[self.data['venue'].mapping[c[0]]]] for c in tops if self.is_venue(c[0])]
            organizations = [[seed, c[0], c[1], self.data['organization'].mapping[c[0]],
                              self.data['organization'].net_x[self.data['organization'].mapping[c[0]]]] for c in tops if
                             self.is_organization(c[0])]

            keys = sel_keys[i]
            keywords = [[seed, c[0], c[1], self.data['keyword'].mapping[c[0]],
                         self.data['keyword'].x[self.data['keyword'].mapping[c[0]]]] for c in keys if
                        self.is_keyword(c[0])]
            entities = [[seed, c[0], c[1], self.data['entity'].mapping[c[0]],
                         self.data['entity'].x[self.data['entity'].mapping[c[0]]]] for c in keys if
                        self.is_entity(c[0])]
            topics = [[seed, c[0], c[1], self.data['topic'].mapping[c[0]],
                       self.data['topic'].x[self.data['topic'].mapping[c[0]]]] for c in keys if self.is_topic(c[0])]
            keywords_net = [[seed, c[0], c[1], self.data['keyword'].mapping[c[0]],
                         self.data['keyword'].net_x[self.data['keyword'].mapping[c[0]]]] for c in keys if
                        self.is_keyword(c[0])]
            entities_net = [[seed, c[0], c[1], self.data['entity'].mapping[c[0]],
                         self.data['entity'].net_x[self.data['entity'].mapping[c[0]]]] for c in keys if
                        self.is_entity(c[0])]
            topics_net = [[seed, c[0], c[1], self.data['topic'].mapping[c[0]],
                       self.data['topic'].net_x[self.data['topic'].mapping[c[0]]]] for c in keys if self.is_topic(c[0])]

            publications_vectors[i] = publications
            publications_net_vectors[i] = publications_net
            datasets_net_vectors[i] = datasets_net
            datasets_vectors[i] = datasets
            entities_vectors[i] = entities
            topics_vectors[i] = topics
            authors_vectors[i] = authors
            venues_vectors[i] = venues
            organizations_vectors[i] = organizations
            keywords_vectors[i] = keywords

            if self.args.split_cores:
                if self.is_publication(seed):
                    if len(datasets) >= self.args.n_cores:
                        ret_cores[i] = datasets[0:self.args.n_cores]
                        ret_net_cores[i] = datasets_net[0:self.args.n_cores]
                    else:
                        ret_cores[i] = datasets + publications
                        ret_cores[i] = ret_cores[i][0:self.args.n_cores]
                        ret_net_cores[i] = datasets_net + publications_net
                        ret_net_cores[i] = ret_net_cores[i][0:self.args.n_cores]
                elif self.is_dataset(seed):
                    if len(publications) >= self.args.n_cores:
                        ret_cores[i] = publications[0:self.args.n_cores]
                        ret_net_cores[i] = publications_net[0:self.args.n_cores]
                    else:
                        ret_cores[i] = publications + datasets
                        ret_cores[i] = ret_cores[i][0:self.args.n_cores]
                        ret_net_cores[i] = publications_net + datasets_net
                        ret_net_cores[i] = ret_net_cores[i][0:self.args.n_cores]
            else:
                ret_cores[i] = datasets + publications
                ret_net_cores[i] = datasets_net + publications_net
                if self.is_dataset(seed):
                    # print('is dat')
                    # print(seed)
                    ret_cores[i] = publications + datasets
                    ret_net_cores[i] = publications_net + datasets_net
                    # for l in ret_cores[i]:
                    #     print(l[1])
                ret_cores[i] = ret_cores[i][:self.args.n_cores]
                ret_net_cores[i] = ret_net_cores[i][:self.args.n_cores]

            ret_keys[i] = topics + entities + keywords
            ret_net_keys[i] = topics_net + entities_net + keywords_net
            ret_tops[i] = venues + organizations + authors

            if len(ret_cores[i]) < self.args.n_cores and len(ret_cores[i]) > 0:
                pad = self.args.n_cores - len(ret_cores[i])
                for _ in range(pad):
                    x = random.choice(ret_cores[i])
                    ret_cores[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.core_dim)])
                    # ret_cores[i].append(x)

            elif len(ret_cores[i]) == 0:
                pad = self.args.n_cores
                for _ in range(pad):
                    ret_cores[i].append(vectors[i])

            if len(ret_net_cores[i]) < self.args.n_cores and len(ret_net_cores[i]) > 0:
                pad = self.args.n_cores - len(ret_net_cores[i])
                for _ in range(pad):
                    x = random.choice(ret_net_cores[i])
                    ret_net_cores[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])
                    # ret_cores[i].append(x)

            elif len(ret_net_cores[i]) == 0:
                pad = self.args.n_cores
                for _ in range(pad):
                    ret_net_cores[i].append(vectors_net[i])

            if len(ret_tops[i]) < self.args.n_top_hubs and len(ret_tops[i]) > 0:
                pad = self.args.n_top_hubs - len(ret_tops[i])
                for _ in range(pad):
                    x = random.choice(ret_tops[i])
                    ret_tops[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])
                    # ret_tops[i].append(x)

            elif len(ret_tops[i]) == 0:
                pad = self.args.n_top_hubs
                for _ in range(pad):
                    ret_tops[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])
            #
            if len(ret_keys[i]) < self.args.n_keys_hubs and len(ret_keys[i]) > 0:
                pad = self.args.n_keys_hubs - len(ret_keys[i])
                for _ in range(pad):
                    x = random.choice(ret_keys[i])
                    ret_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.key_dim)])
                    # ret_keys[i].append(x)

            elif len(ret_keys[i]) == 0:
                pad = self.args.n_keys_hubs
                for _ in range(pad):
                    ret_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.key_dim)])

            if len(ret_net_keys[i]) < self.args.n_keys_hubs and len(ret_net_keys[i]) > 0:
                pad = self.args.n_keys_hubs - len(ret_net_keys[i])
                for _ in range(pad):
                    x = random.choice(ret_net_keys[i])
                    ret_net_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])
                    # ret_keys[i].append(x)

            elif len(ret_net_keys[i]) == 0:
                pad = self.args.n_keys_hubs
                for _ in range(pad):
                    ret_net_keys[i].append([seed, 'pad', -1, -1, torch.zeros(self.args.top_dim)])


            ret_keys[i] = sorted(ret_keys[i], key=lambda x: x[1])
            ret_net_keys[i] = sorted(ret_net_keys[i], key=lambda x: x[1])
            ret_tops[i] = sorted(ret_tops[i], key=lambda x: x[1])


        return vectors, vectors_net, ret_net_cores, ret_cores, ret_net_keys, ret_keys, ret_tops, publications_vectors, datasets_vectors, authors_vectors, entities_vectors, topics_vectors, keywords_vectors, venues_vectors, organizations_vectors

# if __name__ == '__main__':
#     found = False
#     if found:
#         args = get_args()
#         print("------arguments-------")
#         for k, v in vars(args).items():
#             print(k + ': ' + str(v))
#         # random.seed(args.seed)
#         # np.random.seed(args.seed)
#         # torch.manual_seed(args.seed)
#         # torch.cuda.manual_seed_all(args.seed)
#
#         # seeds = pd.read_csv(f'datasets/mes/all/final/publications.csv')['id'].tolist()[0:2]
#         # print(seeds)
#         data = ScholarlyDataset(root='datasets/mes/split_transductive/train/')
#         root = 'datasets/mes/split_transductive/train/'
#         walker_obj = RandomWalkWithRestart(args,data[0],root)
#         walks = walker_obj.create_random_walks()
#         st = time.time()
#         selected_seeds_walks,selected_seeds_cores,selected_seeds_hubs_key,selected_seeds_hubs_top = walker_obj.select_walks(walks)
#         print(time.time()-st)
#         for s in selected_seeds_cores:
#             print(s)
#         a = walker_obj.get_neighbours_vector(selected_seeds_cores,selected_seeds_hubs_key,selected_seeds_hubs_top)

#     #








