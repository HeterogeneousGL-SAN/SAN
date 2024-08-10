import torch
from args_list import get_args
import numpy as np
import random
import loader
import sampler
from loader import *
import os
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score
from utils import EarlyStoppingClass
from sampler import RandomWalkWithRestart
import time
from model import ScHetGNN
from torch_geometric import seed_everything

import torch.nn.functional as F
# in transductive basta fare i path una volta, tanto i nodi li ho sempre visti in ogni set
# in inductive devo farne 3 separati, uno per training, uno per validation e uno per test


"""
Questo è il primo setup, quello originale usato sin dall'inizio



"""
def seed_torch(seed=42):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)



def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0


def ndcg_at_k(true_labels, predictions, k):
    relevance_scores = [1 if item in true_labels else 0 for item in predictions]
    dcg = dcg_at_k(relevance_scores, k)
    idcg = dcg_at_k([1] * len(true_labels), k)  # IDCG assuming all true labels are relevant
    if not idcg:
        return 0
    return dcg / idcg



def cosine_similarity(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_vector1 = np.linalg.norm(emb1)
    norm_vector2 = np.linalg.norm(emb2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity



class Trainer:
    def __init__(self,args,iteration,indices):
        # self.device = 'cpu'
        # if args.train:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iteration = iteration

        print(f'DEVICE {self.device}')
        self.args = args

        train_dataset = loader.ScholarlyDataset(root=f'datasets/{args.dataset}/split_transductive/train/')
        self.train_root = train_dataset.root
        self.dataset = train_dataset[0]
        print(type(self.dataset['dataset'].x))
        print(self.dataset['dataset'].x.shape)
        self.dataset['dataset'].x[indices,:] = 1.0
        print(self.dataset['dataset'].x[indices,:])
        print(self.dataset['dataset'].x)
        self.train_walker = RandomWalkWithRestart(self.args, self.dataset, 'transductive train','train')
        self.validation_walker = RandomWalkWithRestart(self.args, self.dataset, 'transductive train','validation')
        self.test_walker = RandomWalkWithRestart(self.args, self.dataset, 'transductive train','test')

        self.model = ScHetGNN(args).to(self.device)
        self.model.init_weights()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

    def get_walks(self,set_type=None):
        if not os.path.exists(f'./model/data/{self.args.dataset}_inductive_paths_{self.args.inductive_type}_{set_type}.txt'):
            f = open(f'./model/data/{self.args.dataset}_inductive_paths_{self.args.inductive_type}_{set_type}.txt', 'w')
            if set_type == 'train':
                self.train_walker.set_seeds()
                all_walks = self.train_walker.create_random_walks(all=True)
            elif set_type == 'validation':
                self.validation_walker.set_seeds()
                all_walks = self.validation_walker.create_random_walks(all=True)
            elif set_type == 'test':
                self.test_walker.set_seeds()
                all_walks = self.test_walker.create_random_walks(all=True)

            for walk in all_walks:
                f.write(' '.join(walk))
                f.write('\n')
            f.close()
        else:
            if set_type == 'train':
                self.train_walker.set_seeds()
            elif set_type == 'validation':
                self.validation_walker.set_seeds()
            elif set_type == 'test':
                self.test_walker.set_seeds()
            f = open(f'./model/data/{self.args.dataset}_inductive_paths_{self.args.inductive_type}_{set_type}.txt', 'r')
            all_walks = f.readlines()
            all_walks = [w.split() for w in all_walks]
            print(f'walks: {len(all_walks)}')

        assert all_walks != []
        return all_walks


    def trivial_baselines(self):
        """ compute trivial performances"""
        results = json.load(open(f'baselines/trivial/data/{self.args.dataset}/results.json', 'r'))
        mapped_results = {}
        precision, recall, ndcg = 0, 0, 0
        c = 0
        for k, v in results.items():
            mapped_k = self.dataset['publication'].mapping[k]
            mapped_v = [self.dataset['dataset'].mapping[a] for a in v]
            mapped_results[mapped_k] = mapped_v[0:20]
            c += 1
            pred = mapped_v[0:self.args.topk]
            true = self.y_test_true_labels[mapped_k]
            precision += len(list(set(pred).intersection(set(true)))) / self.args.topk
            recall += len(list(set(pred).intersection(set(true)))) / len(true)
            ndcg += ndcg_at_k(true, pred, self.args.topk)
        precision, recall, ndcg = precision/c, recall/c,ndcg/c
        # print(precision / c)
        # print(recall / c)
        # print(ndcg / c)
        reranking_line = 'GOAL precision = {}'.format(precision) + ' recall = {}'.format(
            recall) + ' ndcg = {}'.format(ndcg) + '\n'
        print(reranking_line)
        return mapped_results


    def test(self,test_positive_pd_edges,test_negative_pd_edges,epoch,stringa,best=False):

        # os.listdir('model/checkpoint/transductive/')
        auc_tot = 0
        ap_tot = 0
        rec_tot = 0
        prec_tot = 0
        ndcg_tot = 0
        output_file_path = f"model/checkpoint/inductive/results/{self.args.dataset}/bootstrapped/{self.args.bootstrap}/{self.iteration}_{epoch}_{stringa}.txt"
        # if best:
        #     output_file_path = f"model/checkpoint/inductive/results/{self.args.dataset}/bootstrapped/{self.args.bootstrap}/BEST_{self.iteration}_{epoch}_{stringa}.txt"
        # if epoch + 1 == self.args.epochs:
        #     output_file_path = f"model/checkpoint/inductive/results/{self.args.dataset}/bootstrapped/{self.args.bootstrap}/LAST_{self.iteration}_{epoch}_{stringa}.txt"


        f = open(output_file_path, 'w')
        f.write(stringa + '\n')
        self.model.eval()
        # LINK PREDICTION
        for i in range(10):
            with torch.no_grad():
                y_true_test = np.array([1] * test_positive_pd_edges.size(1) + [0] * test_negative_pd_edges.size(1))


                loss, final_embeddings, pos_source, pos_target, neg_source, neg_target, all_seeds = self.run_minibatch_inductive(
                    self.dataset, 0,
                    test_positive_pd_edges, test_negative_pd_edges,
                    test=True, all_walks=None,set_type='test')

                pos_embeddings_source_ori, neg_embeddings_source_ori = final_embeddings[pos_source], final_embeddings[
                    neg_source]
                pos_embeddings_target_ori, neg_embeddings_target_ori = final_embeddings[pos_target], final_embeddings[
                    neg_target]

                pos_embeddings_source = pos_embeddings_source_ori.view(pos_embeddings_source_ori.size(0), 1,
                                                                       pos_embeddings_source_ori.size(
                                                                           1))  # [batch_size, 1, embed_d]
                neg_embeddings_source = neg_embeddings_source_ori.view(neg_embeddings_source_ori.size(0), 1,
                                                                       neg_embeddings_source_ori.size(
                                                                           1))  # [batch_size, 1, embed_d]
                pos_embeddings_target = pos_embeddings_target_ori.view(pos_embeddings_target_ori.size(0),
                                                                       pos_embeddings_target_ori.size(1),
                                                                       1)  # [batch_size, embed_d, 1]
                neg_embeddings_target = neg_embeddings_target_ori.view(neg_embeddings_target_ori.size(0),
                                                                       neg_embeddings_target_ori.size(1),
                                                                       1)  # [batch_size, embed_d, 1]


                result_positive_matrix = torch.bmm(pos_embeddings_source, pos_embeddings_target)
                result_positive_matrix = torch.sigmoid(result_positive_matrix)
                print(f'result_positive_matrix {result_positive_matrix.shape}')
                result_negative_matrix = torch.bmm(neg_embeddings_source, neg_embeddings_target)
                result_negative_matrix = torch.sigmoid(result_negative_matrix)
                print(f'result_negative_matrix {result_positive_matrix.shape}')

                y_predicted_test = torch.cat([result_positive_matrix.squeeze(), result_negative_matrix.squeeze()])
                y_predicted_test = y_predicted_test.detach().cpu().numpy()
                auc = roc_auc_score(y_true_test, y_predicted_test)
                ap = average_precision_score(y_true_test, y_predicted_test)
                print('Link Prediction Test')
                print('AUC = {}'.format(auc))
                print('AP = {}'.format(ap))


                sources = list(self.y_test_true_labels.keys())
                datasets = list(self.dataset['dataset'].mapping.keys())
                pos_source = [self.dataset['publication'].rev_mapping[j] for j in sources]
                # print(pos_source)
                all_seeds = pos_source + datasets

                self.test_walker.set_seeds(all_seeds)
                # all_walks = [v for k, v in all_walks.items() if k in all_seeds]
                # all_walks = [inner for outer in all_walks for inner in outer]

                all_walks = self.test_walker.create_random_walks(seeds_in=all_seeds)
                selected_seeds_walks, selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top = self.test_walker.select_walks_mp(
                    all_walks)

                seed_vectors,seed_vectors_net,net_cores, cores, net_keys,keys, hubs, _, _, _, _, _, _, _, _ = self.test_walker.get_neighbours_vector(
                    selected_seeds_cores,
                    selected_seeds_hubs_key,
                    selected_seeds_hubs_top)

                pos_source_indices = [all_seeds.index(j) for i, j in enumerate(pos_source)]
                pos_target_indices = [all_seeds.index(j) for i, j in enumerate(datasets)]

                final_embeddings = self.model(seed_vectors, seed_vectors_net,net_cores,cores, net_keys, keys, hubs, core_agg=self.args.core_aggregation,
                                              key_agg=self.args.key_aggregation, top_agg=self.args.top_aggregation,
                                              all_agg=self.args.all_aggregation)

                print(final_embeddings.shape)


                pub_embeddings = F.normalize(final_embeddings[pos_source_indices], p=2, dim=1)
                data_embeddings = F.normalize(final_embeddings[pos_target_indices], p=2, dim=1)

                final_matrix = torch.mm(pub_embeddings, data_embeddings.t())
                print(f'final matrix shape {final_matrix.shape}')
                final_matrix_filtered = torch.mm(pub_embeddings, data_embeddings.t())
                print(f'final matrix shape {final_matrix_filtered.shape}')
                # print(final_matrix_filtered)
                top_values, top_indices = torch.topk(final_matrix, k=len(datasets), dim=1)

                y_test_predicted_labels = {source: [] for source in sources}
                y_test_predicted_labels_norerank = {source: [] for source in sources}
                for i, lista in enumerate(top_indices.tolist()):
                    y_test_predicted_labels_norerank[sources[i]] = lista[0:self.args.topk]

                precision, no_rer_precision = 0, 0
                recall, no_rer_recall = 0, 0
                ndcg, no_rer_ndcg = 0, 0
                run = []
                print(len(sources))
                for source in sources:
                    true = self.y_test_true_labels[source]
                    # pred = y_test_predicted_labels[source][:self.args.topk]

                    # for p in pred:
                    #     run.append(
                    #         f'{self.dataset["publication"].rev_mapping[source]}\tQ0\t{self.dataset["dataset"].rev_mapping[p]}\t{pred.index(p) + 1}\t{pred.index(p) + 1}\tmyrun\n')
                    pred_nonrer = y_test_predicted_labels_norerank[source][:self.args.topk]
                    # precision += len(list(set(pred).intersection(true))) / self.args.topk
                    no_rer_precision += len(list(set(pred_nonrer).intersection(true))) / self.args.topk
                    # recall += len(list(set(pred).intersection(true))) / len(true)
                    no_rer_recall += len(list(set(pred_nonrer).intersection(true))) / len(true)
                    # ndcg += ndcg_at_k(true, pred, self.args.topk)
                    no_rer_ndcg += ndcg_at_k(true, pred_nonrer, self.args.topk)


                print('NO RERANKING')
                print(no_rer_precision / len(sources))
                print(no_rer_recall / len(sources))
                print(no_rer_ndcg / len(sources))
                no_rer_precision, no_rer_recall, no_rer_ndcg  = no_rer_precision / len(sources), no_rer_recall / len(sources), no_rer_ndcg / len(sources)
                prec_tot += no_rer_precision
                rec_tot += no_rer_recall
                ndcg_tot += no_rer_ndcg
                auc_tot += auc
                ap_tot += ap
        prec_tot = prec_tot/10
        rec_tot = rec_tot/10
        ndcg_tot = ndcg_tot/10
        auc_tot = auc_tot/10
        ap_tot = ap_tot/10
        reranking_line = 'AP ={}'.format(ap_tot) + 'AUC ={}'.format(auc_tot) + '\n' + 'STANDARD precision = {}'.format(prec_tot) + ' recall = {}'.format(
            rec_tot) + ' ndcg = {}'.format(ndcg_tot) + '\n'
        print(reranking_line)
        f.write(reranking_line)
        f.close()
        return ap_tot, auc_tot, prec_tot, rec_tot, ndcg_tot


    def run_minibatch_inductive(self,dataset,iteration,positive_edges,negative_edges,test=False, all_walks = None,set_type='train'):

        # seleziono gli indici dell'edge index che mi interessano. Divido per due la batch: voglio ugual numero di archi positivi e negativi
        # i nodi satanno al più il doppio della batchsize perchè ogni arco ha due nodi
        walker_set = self.train_walker
        if set_type == 'train':
            walker_set = self.train_walker
        elif set_type == 'validation':
            walker_set = self.validation_walker
        elif set_type == 'test':
            walker_set = self.test_walker

        if not test:
            batch_positive = positive_edges[:,iteration*self.args.batch_size : (iteration+1)*self.args.batch_size]
            batch_negative = negative_edges[:,iteration*self.args.batch_size : (iteration+1)*self.args.batch_size]

            # if batch_positive.size(1) < args.batch_size:
            #     # padding last minibatch
            #     data_to_add = args.batch_size - batch_positive.size(1)
            #     if self.args.verbose:
            #         print('last batch')
            #         print(batch_positive.shape)
            #         print(batch_negative.shape)
            #         print(data_to_add)
            #     all_pos_edges_but_last_mb = positive_edges[:,:iteration*args.batch_size]
            #     all_neg_edges_but_last_mb = negative_edges[:,:iteration*args.batch_size]
            #     if all_pos_edges_but_last_mb.size(1) > 0:
            #         selected_indices = torch.randint(0, all_pos_edges_but_last_mb.size(1), (data_to_add,))
            #         selected_pos_edges = all_pos_edges_but_last_mb[:, selected_indices]
            #         selected_neg_edges = all_neg_edges_but_last_mb[:, selected_indices]
            #     else:
            #         selected_indices = torch.randint(0, positive_edges.size(1), (data_to_add,))
            #         selected_pos_edges = positive_edges[:, selected_indices]
            #         selected_neg_edges = negative_edges[:, selected_indices]
            #
            #     batch_positive = torch.cat([batch_positive,selected_pos_edges],dim=1)
            #     batch_negative = torch.cat([batch_negative,selected_neg_edges],dim=1)
        else:
            batch_positive = positive_edges
            batch_negative = negative_edges

        positive_sources = batch_positive[0].tolist()
        positive_target = batch_positive[1].tolist()
        mapped_sources = [dataset['publication'].rev_mapping[s] for s in positive_sources]
        mapped_targets = [dataset['dataset'].rev_mapping[s] for s in positive_target]

        positive_seeds = list(set(mapped_sources + mapped_targets))

        negative_sources = batch_negative[0].tolist()
        negative_target = batch_negative[1].tolist()
        mapped_neg_sources = [dataset['publication'].rev_mapping[s] for s in negative_sources]
        mapped_neg_targets = [dataset['dataset'].rev_mapping[s] for s in negative_target]
        negative_seeds = list(set(mapped_neg_sources + mapped_neg_targets))

        all_seeds = list(set(positive_seeds + negative_seeds))
        # print(f'found? vali {[p for p in all_seeds if p == "p_993"]}')
        # print([p for p in positive_seeds if p == 'p_993'])
        # print([p for p in negative_seeds if p == 'p_993'])
        # print('SEARCH')

        # print('p_2849' in all_seeds)
        # print('p_2849' in mapped_neg_sources)
        if self.args.verbose:
            print(f'positive seeds {len(positive_seeds)}')
            print(f'negative seeds {len(negative_seeds)}')
            print(f'all seeds {len(all_seeds)}')

        if all_walks is None:
            all_walks = walker_set.create_random_walks(seeds_in=all_seeds)
            df_test = pd.read_csv('datasets/mes/split_transductive/train/pubdataedges_test_kcore_1.csv')
            test_nodes = df_test['source'].unique().tolist() +df_test['target'].unique().tolist()
            if self.args.inductive_type == 'light':
                test_nodes = df_test['source'].unique().tolist()

            if any(i in test_nodes for i in walker_set.G.nodes()) and set_type != 'test':
                print('ATTENZIONE!!!!!')
        else:
            walker_set.set_seeds(all_seeds)
            all_walks = [v for k, v in all_walks.items() if k in all_seeds]
            all_walks = [inner for outer in all_walks for inner in outer]
            if self.args.verbose:
                print(f'all walks: {len(all_walks)}')
                print(f'all seeds: {len(all_seeds)}')




        pos_source_indices = [all_seeds.index(j) for i,j in enumerate(mapped_sources)]
        pos_target_indices = [all_seeds.index(j) for i,j in enumerate(mapped_targets)]
        neg_source_indices = [all_seeds.index(j) for i,j in enumerate(mapped_neg_sources)]
        neg_target_indices = [all_seeds.index(j) for i,j in enumerate(mapped_neg_targets)]


        # # # seed_vectors: dim = 3: seed, mapped seed, vector
        # # # remaining: dim = 5: seed, id, score, mapped_id, vector
        if self.args.verbose:
            print('start model')

        # args_model = [cores, None, None]
        # if self.args.keep_all:
        #     args_model = [cores, keys, hubs]
        # elif self.args.keep_core_key:
        #     args_model = [cores, keys, None]
        # elif self.args.keep_core:
        #     args_model = [cores, None, None]

        selected_seeds_walks, selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top = walker_set.select_walks_mp(all_walks)
        seed_vectors, seed_vectors_net,net_cores,cores,net_keys,keys,hubs, _, _, _, _, _, _, _, _ = walker_set.get_neighbours_vector(selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top)
        if self.args.verbose:
            print('SEED CHECK')
            print([l[i][0] for i,l in enumerate(cores[0:3])])
            print([l[0] for i,l in enumerate(seed_vectors[0:3])])
            print(all_seeds[0:5])

        final_embeddings = self.model(seed_vectors,seed_vectors_net,net_cores,cores,net_keys,keys,hubs,core_agg=self.args.core_aggregation,key_agg=self.args.key_aggregation,top_agg=self.args.top_aggregation,all_agg=self.args.all_aggregation)
        loss = self.model.cross_entropy_loss(final_embeddings,pos_source_indices,pos_target_indices,neg_source_indices,neg_target_indices)
        if test:
            return loss,final_embeddings,pos_source_indices,pos_target_indices,neg_source_indices,neg_target_indices,all_seeds
        else:
            return loss, final_embeddings


    def run_inductive(self):


        """
        CASE 1: validation and test sets connect nodes already seen in training set
        CASE 2: same but with enriched training set with new edges between nodes not present in validation and test
        CASE 3: original trnsductive split
        """


        # early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(self.args.dataset))

        # first learn node embeddings then use them to the downstream tasks


        edge_label_index_train = self.dataset['publication','cites','dataset'].edge_label_index_train
        edge_label_index_validation = self.dataset['publication','cites','dataset'].edge_label_index_validation
        edge_label_index_test = self.dataset['publication','cites','dataset'].edge_label_index_test

        sources_to_remove = edge_label_index_validation[0].tolist() + edge_label_index_test[0].tolist()
        targets_to_remove = edge_label_index_validation[1].tolist() + edge_label_index_test[1].tolist()
        edge_train_filtered = []
        edge_valid_filtered = []
        neg_edge_train_filtered = []
        neg_edge_valid_filtered = []
        for edge in (edge_label_index_train.t().tolist()):
            if edge[0] not in sources_to_remove and edge[1] not in targets_to_remove and self.args.inductive_type == 'full':
                edge_train_filtered.append(edge)
            elif edge[0] not in sources_to_remove and self.args.inductive_type == 'light':
                edge_train_filtered.append(edge)

        for edge in (edge_label_index_validation.t().tolist() ):
            if edge[0] not in edge_label_index_test[0].tolist() and edge[1] not in edge_label_index_test[1].tolist() and self.args.inductive_type == 'full':
                edge_valid_filtered.append(edge)
            elif edge[0] not in edge_label_index_test[0].tolist() and self.args.inductive_type == 'light':
                edge_valid_filtered.append(edge)

        for edge in (self.dataset['publication', 'cites', 'dataset'].negative_edge_label_index_train.t().tolist()):
            if edge[0] not in sources_to_remove and edge[1] not in targets_to_remove and self.args.inductive_type == 'full':
                neg_edge_train_filtered.append(edge)
            elif edge[0] not in sources_to_remove and self.args.inductive_type == 'light':
                neg_edge_train_filtered.append(edge)
            else:
                # in questo caso campiono per avere un adeguato numero di archi negativi
                found = False
                while not found:
                    source_sel = random.sample([a[0] for a in edge_train_filtered],1)[0]
                    target_sel = random.sample([a[1] for a in edge_train_filtered],1)[0]
                    lista = edge_train_filtered + neg_edge_train_filtered
                    list_found = [a for a in lista if a[0] != source_sel and a[1] != target_sel]
                    if len(list_found) > 0:
                        neg_edge_train_filtered.append([source_sel,target_sel])
                        found = True



        for edge in (self.dataset['publication', 'cites', 'dataset'].negative_edge_label_index_validation.t().tolist() ):
            if edge[0] not in edge_label_index_test[0].tolist() and edge[1] not in edge_label_index_test[1].tolist() and self.args.inductive_type == 'full':
                neg_edge_valid_filtered.append(edge)
            elif edge[0] not in edge_label_index_test[0].tolist() and self.args.inductive_type == 'light':
                neg_edge_valid_filtered.append(edge)
            else:
                # in questo caso campiono per avere un adeguato numero di archi negativi
                found = False
                while not found:
                    source_sel = random.sample([a[0] for a in edge_train_filtered], 1)[0]
                    target_sel = random.sample([a[1] for a in edge_train_filtered], 1)[0]
                    lista = edge_valid_filtered + neg_edge_valid_filtered
                    if len([a for a in lista if a[0] != source_sel and a[1] != target_sel]) > 0:
                        neg_edge_valid_filtered.append([source_sel, target_sel])
                        found = True
        print('VALID?')
        print(len(neg_edge_valid_filtered),len(edge_valid_filtered))
        print(len(neg_edge_train_filtered),len(edge_train_filtered))

        min_len_valid = min([len(edge_valid_filtered),len(neg_edge_valid_filtered)])
        min_len_train = min([len(edge_train_filtered),len(neg_edge_train_filtered)])
        neg_edge_valid_filtered = neg_edge_valid_filtered[0:min_len_valid]
        neg_edge_train_filtered = neg_edge_train_filtered[0:min_len_train]
        edge_train_filtered = edge_train_filtered[0:min_len_train]
        edge_valid_filtered = edge_valid_filtered[0:min_len_valid]
        print('VALID?')
        print(len(neg_edge_valid_filtered),len(edge_valid_filtered))
        print(len(neg_edge_train_filtered),len(edge_train_filtered))
        edge_label_index_train = torch.tensor(edge_train_filtered).t()
        edge_label_index_validation = torch.tensor(edge_valid_filtered).t()
        self.dataset['publication', 'cites', 'dataset'].edge_label_index_validation = edge_label_index_validation
        self.dataset['publication', 'cites', 'dataset'].edge_label_index_train = edge_label_index_train
        negative_edge_label_index_train = torch.tensor(neg_edge_train_filtered).t()
        negative_edge_label_index_validation = torch.tensor(neg_edge_valid_filtered).t()
        self.dataset['publication', 'cites', 'dataset'].negative_edge_label_index_validation = negative_edge_label_index_validation
        self.dataset['publication', 'cites', 'dataset'].negative_edge_label_index_train = negative_edge_label_index_train
        print('Generating negative edges')
        training_positive_pd_edges = edge_label_index_train
        validation_positive_pd_edges = edge_label_index_validation
        test_positive_pd_edges = edge_label_index_test

        # train transductive at different splits
        if self.args.trans:
            sources_to_keep = validation_positive_pd_edges[0].tolist() + test_positive_pd_edges[0].tolist()
            targets_to_keep = validation_positive_pd_edges[1].tolist() + test_positive_pd_edges[1].tolist()
            edge_index_train = [tuple(t) for t in training_positive_pd_edges.t().tolist()]
            edge_index_train_filtered = []

            for tup in edge_index_train:
                if tup[0] in sources_to_keep and tup[1] in targets_to_keep:
                    edge_index_train_filtered.append(tup)

            while len(edge_index_train_filtered) < self.args.trans * len(edge_index_train):
                tuple_sel = random.choice(edge_index_train)
                if tuple_sel not in edge_index_train_filtered:
                    edge_index_train_filtered.append(tuple_sel)

            print(f'trans {trans}')

            edge_index_train_filtered = torch.tensor(edge_index_train_filtered).t()
            self.dataset['publication', 'cites', 'dataset'].edge_index_train = edge_index_train_filtered
            self.dataset['publication', 'cites', 'dataset'].edge_label_index_train = edge_index_train_filtered
            num_edges = edge_index_train_filtered.size(1)
            num_selected = int(num_edges * self.args.trans)
            indices = torch.randperm(num_edges)[:num_selected]
            negatives = self.dataset['publication','cites','dataset'].negative_edge_label_index_train[:, indices]
            self.dataset['publication', 'cites', 'dataset'].negative_edge_label_index_train = negatives


        # if self.args.train:
        # training_negative_pd_edges = loader.cosine_based_negative_samples(edge_index_train,training_positive_pd_edges,source_vectors,target_vectors,similar=False)
        training_negative_pd_edges = self.dataset['publication','cites','dataset'].negative_edge_label_index_train
        validation_negative_pd_edges = self.dataset['publication','cites','dataset'].negative_edge_label_index_validation
        test_negative_pd_edges = self.dataset['publication','cites','dataset'].negative_edge_label_index_test
            # test_negative_pd_edges = loader.cosine_based_negative_samples(edge_index_test,test_positive_pd_edges,source_vectors,target_vectors,similar=False)
        self.y_true_test = np.array([1] * test_positive_pd_edges.size(1) + [0] * test_negative_pd_edges.size(1))
        sources = list(test_positive_pd_edges[0].tolist())
        targets = list(test_positive_pd_edges[1].tolist())
        y_test_true_labels = {source: [] for source in sources}

        for source, target in zip(sources, targets):
            y_test_true_labels[source].append(target)

        self.y_test_true_labels = {k: y_test_true_labels[k] for k in sorted(list(y_test_true_labels.keys()))}
        print('generating trivial baselines')
        self.mapped_results = self.trivial_baselines()
        print('Generating negative edges finished')

        stringa = f'lr-{self.args.lr}_heads-{self.args.heads}_batch-{self.args.batch_size}_cores-{self.args.n_cores}_key-{self.args.n_keys_hubs}_top-{self.args.n_top_hubs}_aggrcore-{self.args.core_aggregation}_aggrkeys-{self.args.key_aggregation}_aggrtop-{self.args.top_aggregation}_allagg-{self.args.all_aggregation}'
        stringa = 'new_'+self.args.dataset + '_'+ stringa+ '_'+ self.args.inductive_type


        save_epoch_path = f'./model/checkpoint/inductive/models/{self.args.dataset}/bootstrapped/{self.args.bootstrap}/{self.iteration}_last_checkpoint_{stringa}_last_epoch.pt'
        save_early_path = f'./model/checkpoint/inductive/models/{self.args.dataset}/bootstrapped/{self.args.bootstrap}/{self.iteration}_best_checkpoint_{stringa}.pt'

        # save_epoch_path = f'./model/checkpoint/inductive/models/{self.args.dataset}/{self.args.enriched}/last_checkpoint_{stringa}_last_epoch.pt'
        # save_early_path = f'./model/checkpoint/inductive/models/{self.args.dataset}/{self.args.enriched}/best_checkpoint_{stringa}.pt'
        early_stopping = EarlyStoppingClass(patience=args.patience, verbose=True, save_epoch_path=save_epoch_path, save_early_path=save_early_path)
        epochs = -1
        print(os.path.exists(save_epoch_path))
        if os.path.exists(save_epoch_path) and not self.args.restart:
            print('LOADING')
            checkpoint = torch.load(save_epoch_path)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epochs = checkpoint['epoch']
                best_score = checkpoint['best_score']
                patience_reached = self.args.patience
                early_stopping = EarlyStoppingClass(patience=patience_reached, verbose=True, save_epoch_path=save_epoch_path,
                                                    save_early_path=save_early_path,best_score=best_score)
            except:
                self.model.load_state_dict(checkpoint)
            print(f'STARTING FROM: epoch {epochs}')
        else:
            print('NEW TRAINING STARTED')


        print('random walks generated')

        if self.args.train:
            for epoch in tqdm.tqdm(range(epochs+1,self.args.epochs),desc="Epoch"):
                # random shuffle before mini-batch training
                t_start = time.time()
                self.model.train()

                # nel training shuffle così ho sempre minibatch diverse
                num_edges_train = edge_label_index_train.size(1)
                num_neg_edges_train = negative_edge_label_index_train.size(1)
                perm_pos_train = torch.randperm(num_edges_train)
                training_positive_pd_edges = training_positive_pd_edges[:, perm_pos_train]
                perm_neg_train = torch.randperm(num_neg_edges_train)
                training_negative_pd_edges = training_negative_pd_edges[:, perm_neg_train]

                num_edges_validation = edge_label_index_validation.size(1)
                num_neg_edges_validation = negative_edge_label_index_validation.size(1)
                perm_pos_validation = torch.randperm(num_edges_validation)
                validation_positive_pd_edges = validation_positive_pd_edges[:, perm_pos_validation]
                perm_neg_validation = torch.randperm(num_neg_edges_validation)
                validation_negative_pd_edges = validation_negative_pd_edges[:, perm_neg_validation]


                train_losses = []
                val_losses = []
                self.model.train()
                for iteration in tqdm.tqdm(range(int(np.ceil(training_positive_pd_edges.size(1) / self.args.batch_size))),
                                           desc="Mini-batch"):
                    loss_train,embeddings = self.run_minibatch_inductive(self.dataset,iteration,training_positive_pd_edges,training_negative_pd_edges,all_walks=None)
                    train_losses.append(loss_train)
                    self.optimizer.zero_grad()
                    loss_train.backward()
                    self.optimizer.step()
                train_loss_final = torch.mean(torch.tensor(train_losses))

                self.model.eval()
                # walks_validation = self.get_walks(set_type='validation')
                # all_walks_vali = {seed: [] for seed in self.validation_walker.G.nodes}
                # for walk in walks_validation:
                #     all_walks_vali[walk[0]].append(walk)

                self.model.eval()
                with torch.no_grad():
                    for iteration in tqdm.tqdm(range(int(np.ceil(validation_positive_pd_edges.size(1) / self.args.batch_size))),
                                               desc="Mini-batch"):
                        # print([p for p in self.validation_walker.G.nodes if p == 'p_993'])
                        # print([p for p in validation_positive_pd_edges[0].tolist() if p == 'p_993'])
                        # print([p for p in validation_negative_pd_edges[0].tolist() if p == 'p_993'])
                        loss_validation,_ = self.run_minibatch_inductive(self.dataset,iteration,validation_positive_pd_edges,validation_negative_pd_edges,all_walks=None,set_type='validation')

                        val_losses.append(loss_validation)
                    t_end = time.time()
                    val_loss_final = torch.mean(torch.tensor(val_losses))
                    print('Epoch {:05d} |Train loss {:.4f} | Val Loss {:.4f} | Time(s) {:.4f}'.format(
                        epoch,train_loss_final.item(), val_loss_final.item(), t_end - t_start))
                    # early stopping

                    early_stopping(val_loss_final, self.model,self.optimizer,epoch)
                    # if early_stopping.best_epoch == epoch and self.args.inductive_type == 'light':
                    #     self.test(test_positive_pd_edges, test_negative_pd_edges, epoch,stringa,best=True)
                    # if epoch + 1 == self.args.epochs and self.args.inductive_type == 'light':
                    #     self.test(test_positive_pd_edges, test_negative_pd_edges, epoch,stringa)

                    # if early_stopping.save or ((epoch + 1) % 10 == 0 and epoch != 0):
                    #     print(f'compute test for epoch: {epoch}')
                    #     self.test(test_positive_pd_edges, test_negative_pd_edges, epoch, stringa)

                    if early_stopping.early_stop:
                        # self.test(test_positive_pd_edges, test_negative_pd_edges, epoch, stringa)
                        print('Early stopping!')
                        break



        if not self.args.train:
            stringa = f'lr-{self.args.lr}_heads-{self.args.heads}_batch-{self.args.batch_size}_cores-{self.args.n_cores}_key-{self.args.n_keys_hubs}_top-{self.args.n_top_hubs}_aggrcore-{self.args.core_aggregation}_aggrkeys-{self.args.key_aggregation}_aggrtop-{self.args.top_aggregation}_allagg-{self.args.all_aggregation}'
            stringa = 'new_'+self.args.dataset + '_' + stringa
            ap_tot, auc_tot, prec_tot, rec_tot, ndcg_tot = self.test(test_positive_pd_edges, test_negative_pd_edges, 'testepoch', stringa)
            return ap_tot, auc_tot, prec_tot, rec_tot, ndcg_tot


if __name__ == '__main__':
    args = get_args()

    print("------arguments-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    for boot in [False]:
        args = get_args()

        args.hetgnn = boot
        train_dataset = loader.ScholarlyDataset(root=f'datasets/{args.dataset}/split_transductive/train/')
        dataset = train_dataset[0]
        datasets = dataset['publication','cites','dataset'].edge_label_index_test[1].tolist() + dataset['publication','cites','dataset'].negative_edge_label_index_test[1].tolist()
        data_num = dataset['dataset'].x.shape[0]
        print(f'the number of dataset is: {data_num}')
        seed_torch(args.random_seed)
        indices_dict = {}

        if args.hetgnn:
            args.core_aggregation = 'lstm'
            args.key_aggregation = 'lstm'
            args.top_aggregation = 'lstm'
            args.all_aggregation = 'mh-attention'
            args.heads = 1
            args.epochs = 100
        else:
            args.batch_size = 4096
            args.epochs = 200
    
    
        if args.train and args.dataset == 'mes':
                num_data =
                args.bootstrap = str(b)
                outpath = f'model/checkpoint/inductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices{args.inductive_type}_{args.batch_size}.json'
                if args.hetgnn:
                    outpath = f'model/checkpoint/inductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices{args.inductive_type}_{args.batch_size}_hetgnn.json'

                for i in range(0,1):
                    print(f'ITERATION: {i}')
        
                    if i > 0 or os.path.exists(outpath):
                        f = open(outpath,'r')
                        indices_dict = json.load(f)
                        f.close()
                    number_perm = int((int(args.bootstrap)/100)*2949)
                    indices = random.sample(range(2949), number_perm)
        
                    if args.inductive_type == 'full':
                        indices = [j for j in indices if j not in datasets]
                        while len(indices) < number_perm:
                            z = random.sample(range(2949), int(number_perm - len(indices)))
                            indices.extend(z)
                            indices = [j for j in indices if j not in datasets]
                    indices_dict[f'iter_{str(i)}'] = indices
                    g = open(outpath, 'w')
                    json.dump(indices_dict,g)
                    g.close()
        
        
                    trainer = Trainer(args,i,indices)
                    trainer.run_inductive()
                    g.close()
        elif args.dataset == 'pubmed':
            outpath = f'model/checkpoint/inductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices{args.inductive_type}_{args.batch_size}.json'
            if args.hetgnn:
                outpath = f'model/checkpoint/inductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices{args.inductive_type}_{args.batch_size}_hetgnn.json'

            for i in range(int(args.iteration), int(args.iteration+1)):
                print(f'ITERATION: {i}')

                if i > 0 or os.path.exists(outpath):
                    f = open(outpath, 'r')
                    indices_dict = json.load(f)
                    f.close()
                number_perm = int((int(args.bootstrap) / 100) * data_num)
                indices = random.sample(range(data_num), number_perm)

                if args.inductive_type == 'full':
                    indices = [j for j in indices if j not in datasets]
                    while len(indices) < number_perm:
                        z = random.sample(range(data_num), int(number_perm - len(indices)))
                        indices.extend(z)
                        indices = [j for j in indices if j not in datasets]
                indices_dict[f'iter_{str(i)}'] = indices
                g = open(outpath, 'w')
                json.dump(indices_dict, g)
                g.close()
                trner(args, i, indices)
                trainer.run_inductive()
                g.close()
                
        if args.test:
            for ty in ['light','full']:
                for b in [25,50,75]:
                    # args = get_args()
                    args.hetgnn = boot
                    args.inductive_type = ty
                    args.bootstrap = b
                    outpath = f'model/checkpoint/inductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices{args.inductive_type}_{args.batch_size}.json'
                    if args.hetgnn:
                        outpath = f'model/checkpoint/inductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices{args.inductive_type}_{args.batch_size}_hetgnn.json'

                    ap, auc, prec, rec, ndcg = [], [], [], [], []
                    for i in range(0, 10):
                        f = open(outpath,   'r')
                        indices_dict = json.load(f)
                        indices = indices_dict[f'iter_{str(i)}']
                        trainer = Trainer(args, i, indices)
                        ap_tot, auc_tot, prec_tot, rec_tot, ndcg_tot = trainer.run_inductive()
                        ap.append(ap_tot)
                        auc.append(auc_tot)
                        prec.append(prec_tot)
                        rec.append(rec_tot)
                        ndcg.append(ndcg_tot)
    
                    output_file_path = f"model/checkpoint/inductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/recap_{args.inductive_type}_{args.batch_size}_{args.epochs}.txt"
                    if args.hetgnn:
                        output_file_path = f"model/checkpoint/inductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/recap_{args.inductive_type}_{args.batch_size}_{args.epochs}_hetgnn.txt"
    
                    f = open(output_file_path, 'w')
                    f.write(f"AP {sum(ap) / 10} AUC {sum(auc) / 10}\n P {sum(prec) / 10} R {sum(rec) / 10} NDCG {sum(ndcg) / 10}\n")
                    f.write(
                        f"AP {np.std(ap) / np.sqrt(10)} AUC {np.std(auc) / np.sqrt(10)}\n P {np.std(prec) / np.sqrt(10)} R {np.std(rec) / np.sqrt(10)} NDCG {np.std(ndcg) / np.sqrt(10)}")
                    f.close()
    
    
    
