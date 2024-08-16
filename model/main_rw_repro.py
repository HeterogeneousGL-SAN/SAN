import torch
from args_list import get_args
import numpy as np
import random
import os

import loader_repro as loader
import sampler_repro as sampler
from loader_repro import *
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score
from utils import EarlyStoppingClass
from sampler_repro import RandomWalkWithRestart
import time
from model_repro import ScHetGNN
from torch_geometric import seed_everything
seed_everything(42)
import torch.nn.functional as F
# in transductive basta fare i path una volta, tanto i nodi li ho sempre visti in ogni set
# in inductive devo farne 3 separati, uno per training, uno per validation e uno per test


"""
Questo è il primo setup, quello originale usato sin dall'inizio



"""
# seed_everything(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# random.seed(42)
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark = False
args = get_args()
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

seed_torch()



# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True



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
    def __init__(self,args):
        # self.device = 'cpu'
        # if args.train:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f'DEVICE {self.device}')
        self.args = args

        train_dataset = loader.ScholarlyDataset(root=f'datasets/{args.dataset}/split_transductive/train/')
        self.train_root = train_dataset.root
        self.dataset = train_dataset[0]
        if args.no_metadata:
            print(type(self.dataset['dataset'].x))
            print(self.dataset['dataset'].x.shape)
            self.dataset['dataset'].x = torch.ones(self.dataset['dataset'].x.shape[0],384)
        self.walker = RandomWalkWithRestart(self.args, self.dataset, 'transductive train')
        self.model = ScHetGNN(args).to(self.device)
        self.model.init_weights()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

    def get_walks(self):
        if not os.path.exists(f'./model/data/{self.args.dataset}_transductive_paths.txt'):
            f = open(f'./model/data/{self.args.dataset}_transductive_paths.txt', 'w')

            all_walks = self.walker.create_random_walks(all=True)
            for walk in all_walks:
                f.write(' '.join(walk))
                f.write('\n')
            f.close()
        else:
            self.walker.set_seeds()
            f = open(f'./model/data/{self.args.dataset}_test_walks.txt', 'r')
            all_walks = f.readlines()
            all_walks = [w.split() for w in all_walks]
            print(f'walks: {len(all_walks)}')
        assert all_walks != []
        return all_walks

    def get_test_walks(self):
        if not os.path.exists(f'./model/data/{self.args.dataset}_test_walks.txt'):
            f = open(f'./model/data/{self.args.dataset}_test_walks.txt', 'w')

            all_walks = self.walker.create_random_walks(all=True)
            for walk in all_walks:
                f.write(' '.join(walk))
                f.write('\n')
            f.close()
        else:
            self.walker.set_seeds()
            f = open(f'./model/data/{self.args.dataset}_test_walks.txt', 'r')
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

        reranking_line = 'GOAL precision = {}'.format(precision) + ' recall = {}'.format(
            recall) + ' ndcg = {}'.format(ndcg) + '\n'
        print(reranking_line)
        return mapped_results


    def test(self,test_positive_pd_edges,test_negative_pd_edges,epoch,stringa,best=False):
        with torch.no_grad():

            output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/{self.args.enriched}/RESULTS_{epoch}_{stringa}.txt"

            if self.args.no_metadata:
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/{self.args.enriched}/RESULTS_{epoch}_{stringa}_no_metadata.txt"

            print(output_file_path)
            if best:
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/{self.args.enriched}/BEST_{epoch}_{stringa}.txt"
            if isinstance(epoch,int) and epoch + 1 == self.args.epochs:
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/{self.args.enriched}/LAST_{epoch}_{stringa}.txt"

            if self.args.eval_lr:
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/lr_{self.args.lr}/{epoch}_{stringa}.txt"
            elif self.args.eval_batch:
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/batch_{self.args.batch_size}/{epoch}_{stringa}.txt"
            elif self.args.eval_combine_aggregation:
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/aggr_all_{self.args.all_aggregation}/{epoch}_{stringa}.txt"
            elif self.args.eval_aggregation:
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/aggr_{self.args.core_aggregation}/{epoch}_{stringa}.txt"
            elif self.args.eval_neigh:
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/neigh_{self.args.n_cores}_{self.args.n_keys_hubs}_{self.args.n_top_hubs}/{epoch}_{stringa}.txt"
            elif self.args.eval_heads:
                output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/ablation/heads_{self.args.heads}/{epoch}_{stringa}.txt"
            f = open(output_file_path, 'w')
            f.write(stringa + '\n')
            self.model.eval()
            auc_tot = 0
            ap_tot = 0
            rec_tot = 0
            prec_tot = 0
            ndcg_tot = 0
            if not os.path.exists(f'./model/data/test_walks/{self.args.dataset}_best_test_paths.txt'):
                range_s = 30
                if self.args.dataset =='pubmed':
                    range_s = 1
                for iter in range(range_s):
                    print('eval round: ',str(iter))
                    sources = list(self.y_test_true_labels.keys())
                    neg_sources = list(test_negative_pd_edges[0].tolist())
                    datasets = list(self.dataset['dataset'].mapping.keys())
                    pos_source = [self.dataset['publication'].rev_mapping[j] for j in sources]
                    neg_source = [self.dataset['publication'].rev_mapping[j] for j in neg_sources]
                    # print(pos_source)
                    all_seeds = pos_source + datasets
                    self.walker.set_seeds(all_seeds)
                    if not os.path.exists(f'./model/data/test_walks/{self.args.dataset}_{iter}_test_paths.txt'):
                        ff = open(f'./model/data/test_walks/{self.args.dataset}_{iter}_test_paths.txt', 'w')
                        all_walks = self.walker.create_random_walks(seeds_in=all_seeds + neg_source)
                        self.walker.set_seeds(all_seeds)
                        for walk in all_walks:
                            ff.write(' '.join(walk))
                            ff.write('\n')
                        ff.close()
                    else:
                        ff = open(f'./model/data/test_walks/{self.args.dataset}_{iter}_test_paths.txt', 'r')
                        all_walks = ff.readlines()
                        ff.close()
                        walks = [w.split() for w in all_walks]
                        self.walker.set_seeds(all_seeds)
                        all_walks = {seed: [] for seed in self.walker.G.nodes if
                                     self.walker.is_publication(seed) or self.walker.is_dataset(seed)}
                        for walk in walks:
                            all_walks[walk[0]].append(walk)
                        all_walks = {k: v for k, v in all_walks.items() if len(v) > 0}
                        all_walks = [v for k, v in all_walks.items() if k in all_seeds]
                        all_walks = [inner for outer in all_walks for inner in outer]
                    selected_seeds_walks, selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top = self.walker.select_walks_mp(all_walks)
                    seed_vectors,seed_vectors_net, net_cores, cores, net_keys, keys, hubs, _, _, _, _, _, _, _, _ = self.walker.get_neighbours_vector(
                        selected_seeds_cores,
                        selected_seeds_hubs_key,
                        selected_seeds_hubs_top)


                    pos_source_indices = [all_seeds.index(j) for i, j in enumerate(pos_source)]
                    pos_target_indices = [all_seeds.index(j) for i, j in enumerate(datasets)]

                    final_embeddings = self.model(seed_vectors,seed_vectors_net, net_cores, cores, net_keys, keys, hubs, core_agg=self.args.core_aggregation,
                                                  key_agg=self.args.key_aggregation, top_agg=self.args.top_aggregation,
                                                  all_agg=self.args.all_aggregation)

                    # Calcola il prodotto scalare tra gli embeddings normalizzati
                    pub_embeddings = F.normalize(final_embeddings[pos_source_indices], p=2, dim=1)
                    data_embeddings = F.normalize(final_embeddings[pos_target_indices], p=2, dim=1)
                    final_matrix = torch.mm(pub_embeddings, data_embeddings.t())
                    print(f'final matrix shape {final_matrix.shape}')
                    final_matrix_filtered = torch.mm(pub_embeddings, data_embeddings.t())
                    print(f'final matrix shape {final_matrix_filtered.shape}')

                    top_values, top_indices = torch.topk(final_matrix, k=len(datasets), dim=1)

                    y_test_predicted_labels_norerank = {source: [] for source in sources}
                    # mapped_results = self.trivial_baselines()
                    for i, lista in enumerate(top_indices.tolist()):
                        y_test_predicted_labels_norerank[sources[i]] = lista[0:self.args.topk]

                    precision, no_rer_precision = 0, 0
                    recall, no_rer_recall = 0, 0
                    ndcg, no_rer_ndcg = 0, 0
                    run = []
                    print(len(sources))


                    # LINK PREDICTION
                    y_true_test = np.array([1] * test_positive_pd_edges.size(1) + [0] * test_negative_pd_edges.size(1))
                    all_walks = None
                    if os.path.exists(f'./model/data/test_walks/{self.args.dataset}_{iter}_test_paths.txt'):
                        g = open(f'./model/data/test_walks/{self.args.dataset}_{iter}_test_paths.txt', 'r')
                        all_walks = g.readlines()
                        walks = [w.split() for w in all_walks]
                        all_walks = {seed: [] for seed in self.walker.G.nodes if self.walker.is_publication(seed) or self.walker.is_dataset(seed)}

                        for walk in walks:
                            all_walks[walk[0]].append(walk)
                        all_walks = {k:v for k,v in all_walks.items() if len(v) > 0}


                    loss, final_embeddings, pos_source, pos_target, neg_source, neg_target, all_seeds = self.run_minibatch_transductive(
                        self.dataset, 0,
                        test_positive_pd_edges, test_negative_pd_edges,
                        test=True, all_walks=all_walks)

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
                    result_negative_matrix = torch.bmm(neg_embeddings_source, neg_embeddings_target)
                    result_negative_matrix = torch.sigmoid(result_negative_matrix)

                    y_predicted_test = torch.cat([result_positive_matrix.squeeze(), result_negative_matrix.squeeze()])
                    y_predicted_test = y_predicted_test.detach().cpu().numpy()
                    auc = roc_auc_score(y_true_test, y_predicted_test)
                    ap = average_precision_score(y_true_test, y_predicted_test)
                    print('Link Prediction Test')
                    print('AUC = {}'.format(auc))
                    print('AP = {}'.format(ap))

                    for topk in [1,5,10]:
                        for source in sources:
                            true = self.y_test_true_labels[source]
                            # pred = y_test_predicted_labels[source][:self.args.topk]
                            # print(true)
                            # print(pred)

                            # for p in pred:
                            #     run.append(f'{self.dataset["publication"].rev_mapping[source]}\tQ0\t{self.dataset["dataset"].rev_mapping[p]}\t{pred.index(p)+1}\t{pred.index(p)+1}\tmyrun\n')
                            pred_nonrer = y_test_predicted_labels_norerank[source][:topk]
                            # precision += len(list(set(pred).intersection(true))) / self.args.topk
                            no_rer_precision += len(list(set(pred_nonrer).intersection(true))) / topk
                            # recall += len(list(set(pred).intersection(true))) / len(true)
                            no_rer_recall += len(list(set(pred_nonrer).intersection(true))) / len(true)
                            # ndcg += ndcg_at_k(true, pred, self.args.topk)
                            no_rer_ndcg += ndcg_at_k(true, pred_nonrer, topk)

                        print('NO RERANKING')
                        print(no_rer_precision / len(sources))
                        print(no_rer_recall / len(sources))
                        print(no_rer_ndcg / len(sources))
                        no_rer_precision, no_rer_recall, no_rer_ndcg = no_rer_precision / len(sources), no_rer_recall / len(
                            sources), no_rer_ndcg / len(sources)
                        if topk == 5:
                            prec_tot += no_rer_precision
                            rec_tot += no_rer_recall
                            ndcg_tot += no_rer_ndcg
                            auc_tot += auc
                            ap_tot += ap
                        reranking_line = '\niteration ={}'.format(str(iter)) + '\ntopk ={}'.format(str(topk)) + '\nAP ={}'.format(ap) + ' AUC ={}'.format(
                            auc) + '\n' + 'STANDARD precision = {}'.format(no_rer_precision) + ' recall = {}'.format(
                            no_rer_recall) + ' ndcg = {}'.format(no_rer_ndcg) + '\n\n'
                        f.write(reranking_line)

                prec_tot = prec_tot / range_s
                rec_tot = rec_tot / range_s
                ndcg_tot = ndcg_tot / range_s
                auc_tot = auc_tot / range_s
                ap_tot = ap_tot / range_s

                reranking_line = 'TOTALE RUNS AP ={}'.format(ap_tot) + 'AUC ={}'.format(
                    auc_tot) + '\n' + 'STANDARD precision = {}'.format(prec_tot) + ' recall = {}'.format(
                    rec_tot) + ' ndcg = {}'.format(ndcg_tot) + '\n'
                print(reranking_line)
                f.write(reranking_line)
                f.close()
            else:
                # LINK PREDICTION
                y_true_test = np.array([1] * test_positive_pd_edges.size(1) + [0] * test_negative_pd_edges.size(1))
                all_walks = None
                if os.path.exists(f'./model/data/test_walks/{self.args.dataset}_best_test_paths.txt'):
                    g = open(f'./model/data/test_walks/{self.args.dataset}_best_test_paths.txt', 'r')
                    all_walks = g.readlines()
                    walks = [w.split() for w in all_walks]
                    all_walks = {seed: [] for seed in self.walker.G.nodes if
                                 self.walker.is_publication(seed) or self.walker.is_dataset(seed)}

                    for walk in walks:
                        all_walks[walk[0]].append(walk)
                    all_walks = {k: v for k, v in all_walks.items() if len(v) > 0}

                loss, final_embeddings, pos_source, pos_target, neg_source, neg_target, all_seeds = self.run_minibatch_transductive(
                    self.dataset, 0,
                    test_positive_pd_edges, test_negative_pd_edges,
                    test=True, all_walks=all_walks)

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
                result_negative_matrix = torch.bmm(neg_embeddings_source, neg_embeddings_target)
                result_negative_matrix = torch.sigmoid(result_negative_matrix)

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
                ff = open(f'./model/data/test_walks/{self.args.dataset}_best_test_paths.txt', 'r')
                all_walks = ff.readlines()
                walks = [w.split() for w in all_walks]
                self.walker.set_seeds(all_seeds)
                all_walks = {seed: [] for seed in self.walker.G.nodes if self.walker.is_publication(seed) or self.walker.is_dataset(seed)}
                for walk in walks:
                    all_walks[walk[0]].append(walk)
                all_walks = {k:v for k,v in all_walks.items() if len(v) > 0}
                all_walks = [v for k, v in all_walks.items() if k in all_seeds]
                all_walks = [inner for outer in all_walks for inner in outer]


                selected_seeds_walks, selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top = self.walker.select_walks_mp(
                    all_walks)
                seed_vectors, seed_vectors_net, net_cores, cores, net_keys, keys, hubs, _, _, _, _, _, _, _, _ = self.walker.get_neighbours_vector(
                    selected_seeds_cores,
                    selected_seeds_hubs_key,
                    selected_seeds_hubs_top)

                pos_source_indices = [all_seeds.index(j) for i, j in enumerate(pos_source)]
                pos_target_indices = [all_seeds.index(j) for i, j in enumerate(datasets)]

                final_embeddings = self.model(seed_vectors, seed_vectors_net, net_cores, cores, net_keys, keys, hubs,
                                              core_agg=self.args.core_aggregation,
                                              key_agg=self.args.key_aggregation, top_agg=self.args.top_aggregation,
                                              all_agg=self.args.all_aggregation)

                # Calcola il prodotto scalare tra gli embeddings normalizzati
                pub_embeddings = F.normalize(final_embeddings[pos_source_indices], p=2, dim=1)
                data_embeddings = F.normalize(final_embeddings[pos_target_indices], p=2, dim=1)
                final_matrix = torch.mm(pub_embeddings, data_embeddings.t())
                print(f'final matrix shape {final_matrix.shape}')
                final_matrix_filtered = torch.mm(pub_embeddings, data_embeddings.t())
                print(f'final matrix shape {final_matrix_filtered.shape}')

                top_values, top_indices = torch.topk(final_matrix, k=len(datasets), dim=1)

                y_test_predicted_labels_norerank = {source: [] for source in sources}
                # mapped_results = self.trivial_baselines()
                for i, lista in enumerate(top_indices.tolist()):
                    y_test_predicted_labels_norerank[sources[i]] = lista[0:self.args.topk]


                for topk in [1,5,10]:
                    precision, no_rer_precision = 0, 0
                    recall, no_rer_recall = 0, 0
                    ndcg, no_rer_ndcg = 0, 0
                    run = []
                    print(len(sources))

                    for source in sources:
                        true = self.y_test_true_labels[source]
                        # pred = y_test_predicted_labels[source][:self.args.topk]
                        # print(true)
                        # print(pred)

                        # for p in pred:
                        #     run.append(f'{self.dataset["publication"].rev_mapping[source]}\tQ0\t{self.dataset["dataset"].rev_mapping[p]}\t{pred.index(p)+1}\t{pred.index(p)+1}\tmyrun\n')
                        pred_nonrer = y_test_predicted_labels_norerank[source][:topk]
                        # precision += len(list(set(pred).intersection(true))) / self.args.topk
                        no_rer_precision += len(list(set(pred_nonrer).intersection(true))) / topk
                        # recall += len(list(set(pred).intersection(true))) / len(true)
                        no_rer_recall += len(list(set(pred_nonrer).intersection(true))) / len(true)
                        # ndcg += ndcg_at_k(true, pred, self.args.topk)
                        no_rer_ndcg += ndcg_at_k(true, pred_nonrer, topk)

                    print(f'NO RERANKING topk {topk}')
                    print(no_rer_precision / len(sources))
                    print(no_rer_recall / len(sources))
                    print(no_rer_ndcg / len(sources))
                    no_rer_precision, no_rer_recall, no_rer_ndcg = no_rer_precision / len(sources), no_rer_recall / len(
                        sources), no_rer_ndcg / len(sources)

                    reranking_line = '\n topk ={}'.format(str(topk)) + '\nAP ={}'.format(ap) + 'AUC ={}'.format(
                        auc) + '\n' + 'STANDARD precision = {}'.format(no_rer_precision) + ' recall = {}'.format(
                        no_rer_recall) + ' ndcg = {}'.format(no_rer_ndcg) + '\n\n'
                    f.write(reranking_line)

            # prec_tot = prec_tot / 10
            # rec_tot = rec_tot / 10
            # ndcg_tot = ndcg_tot / 10
            # auc_tot = auc_tot / 10
            # ap_tot = ap_tot / 10
            #
            # reranking_line = 'TOTALE RUNS AP ={}'.format(ap_tot) + 'AUC ={}'.format(
            #     auc_tot) + '\n' + 'STANDARD precision = {}'.format(prec_tot) + ' recall = {}'.format(
            #     rec_tot) + ' ndcg = {}'.format(ndcg_tot) + '\n'
            # print(reranking_line)
            # f.write(reranking_line)
            #
            # for r in run:
            #     f.write(r)
            f.close()



    def run_minibatch_transductive(self,dataset,iteration,positive_edges,negative_edges,test=False, all_walks = None):

        # seleziono gli indici dell'edge index che mi interessano. Divido per due la batch: voglio ugual numero di archi positivi e negativi
        # i nodi satanno al più il doppio della batchsize perchè ogni arco ha due nodi
        if not test:
            batch_positive = positive_edges[:,iteration*self.args.batch_size : (iteration+1)*self.args.batch_size]
            batch_negative = negative_edges[:,iteration*self.args.batch_size : (iteration+1)*self.args.batch_size]

        else:
            batch_positive = positive_edges
            batch_negative = negative_edges

        positive_sources = batch_positive[0].tolist()
        positive_target = batch_positive[1].tolist()
        mapped_sources = [dataset['publication'].rev_mapping[s] for s in positive_sources]
        mapped_targets = [dataset['dataset'].rev_mapping[s] for s in positive_target]
        positive_seeds = sorted(list(set(mapped_sources + mapped_targets)))

        negative_sources = batch_negative[0].tolist()
        negative_target = batch_negative[1].tolist()
        mapped_neg_sources = [dataset['publication'].rev_mapping[s] for s in negative_sources]
        mapped_neg_targets = [dataset['dataset'].rev_mapping[s] for s in negative_target]
        negative_seeds = sorted(list(set(mapped_neg_sources + mapped_neg_targets)))

        all_seeds = sorted(list(set(positive_seeds + negative_seeds)))

        if self.args.verbose:
            print(f'positive seeds {len(positive_seeds)}')
            print(f'negative seeds {len(negative_seeds)}')
            print(f'all seeds {len(all_seeds)}')

        if all_walks is None:
            all_walks = self.walker.create_random_walks(seeds_in=all_seeds)
        else:
            self.walker.set_seeds(all_seeds)
            all_walks = [v for k, v in all_walks.items() if k in all_seeds]
            all_walks = [inner for outer in all_walks for inner in outer]


        # print(selected_seeds_walks[0])
        # print(selected_seeds_cores[0])
        # for k,s in enumerate(seed_vectors):
            # if s[0] == 'd_1' or s[0] == 'p_3552':
            #     print(f'seed {s[0:-2]}')
            #     print(f'cores {[l[:-2] for l in cores[k]]}')
            #     print(f'cores {[l[:-2] for l in keys[k]]}')
        # print(len(cores))
        # print(len(cores[0]))
        # print(cores[0])


        pos_source_indices = [all_seeds.index(j) for i,j in enumerate(mapped_sources)]
        pos_target_indices = [all_seeds.index(j) for i,j in enumerate(mapped_targets)]
        neg_source_indices = [all_seeds.index(j) for i,j in enumerate(mapped_neg_sources)]
        neg_target_indices = [all_seeds.index(j) for i,j in enumerate(mapped_neg_targets)]


        # # # seed_vectors: dim = 3: seed, mapped seed, vector
        # # # remaining: dim = 5: seed, id, score, mapped_id, vector
        if self.args.verbose:
            print('start model')

        selected_seeds_walks, selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top = self.walker.select_walks_mp(all_walks)
        seed_vectors,seed_vectors_net,net_cores,cores,net_keys,keys,hubs, _, _, _, _, _, _, _, _ = self.walker.get_neighbours_vector(selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top)
        final_embeddings = self.model(seed_vectors,seed_vectors_net,net_cores, cores,net_keys,keys,hubs,core_agg=self.args.core_aggregation,key_agg=self.args.key_aggregation,top_agg=self.args.top_aggregation,all_agg=self.args.all_aggregation)
        loss = self.model.cross_entropy_loss(final_embeddings,pos_source_indices,pos_target_indices,neg_source_indices,neg_target_indices)
        if test:
            return loss,final_embeddings,pos_source_indices,pos_target_indices,neg_source_indices,neg_target_indices,all_seeds
        else:
            return loss, final_embeddings


    def run_transductive(self):


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
        # self.mapped_results = self.trivial_baselines()
        # print('Generating negative edges finished')

        stringa = f'lr-{self.args.lr}_heads-{self.args.heads}_batch-{self.args.batch_size}_cores-{self.args.n_cores}_key-{self.args.n_keys_hubs}_top-{self.args.n_top_hubs}_aggrcore-{self.args.core_aggregation}_aggrkeys-{self.args.key_aggregation}_aggrtop-{self.args.top_aggregation}_allagg-{self.args.all_aggregation}'
        stringa = 'test_gen_new_'+self.args.dataset + '_'+ stringa
        if self.args.no_metadata:
            stringa = 'NO_METADATA_new_' + stringa
            
        save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/{self.args.enriched}/last_checkpoint_{stringa}_last_epoch.pt'
        save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/{self.args.enriched}/best_checkpoint_{stringa}.pt'
        if self.args.eval_lr:
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/lr_{self.args.lr}/checkpoint_{stringa}_last_epoch.pt'
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/lr_{self.args.lr}/checkpoint_{stringa}.pt'
        elif self.args.eval_batch:
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/batch_{self.args.batch_size}/checkpoint_{stringa}_last_epoch.pt'
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/batch_{self.args.batch_size}/checkpoint_{stringa}.pt'
        elif self.args.eval_combine_aggregation:
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/aggr_all_{self.args.all_aggregation}/checkpoint_{stringa}.pt'
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/aggr_all_{self.args.all_aggregation}/checkpoint_{stringa}_last_epoch.pt'
        elif self.args.eval_aggregation:
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/aggr_{self.args.core_aggregation}/checkpoint_{stringa}_last_epoch.pt'
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/aggr_{self.args.core_aggregation}/checkpoint_{stringa}.pt'
        elif self.args.eval_neigh:
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/neigh_{self.args.n_cores}_{self.args.n_keys_hubs}_{self.args.n_top_hubs}/checkpoint_{stringa}_last_epoch.pt'
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/neigh_{self.args.n_cores}_{self.args.n_keys_hubs}_{self.args.n_top_hubs}/checkpoint_{stringa}.pt'
        elif self.args.eval_heads:
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/heads_{self.args.heads}/checkpoint_{stringa}_last_epoch.pt'
            save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/ablation/heads_{self.args.heads}/checkpoint_{stringa}.pt'


        early_stopping = EarlyStoppingClass(patience=args.patience, verbose=True, save_epoch_path=save_epoch_path, save_early_path=save_early_path)
        epochs = -1
        print(os.path.exists(save_epoch_path))
        if args.test:
            m = 'san'
            if args.hetgnn:
                m = 'hetgnn'
            # mh - attention_aggrkeys - mh - attention_aggrtop - mh - attention_allagg - concat_last_epoch_epoch_99.pt
            save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/enriched_all/{m}/metadata/last_checkpoint_{stringa}_last_epoch_epoch_149.pt'
            if args.no_metadata:
                save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/enriched_all/{m}/no_metadata/last_checkpoint_{stringa}_last_epoch_epoch_149.pt'
        print(save_epoch_path)
        if os.path.exists(save_epoch_path) and not self.args.restart:
            print('LOADING')

            # path = save_epoch_path
            # checkpoint = torch.load(path)
            # self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epochs = checkpoint['epoch']
            # best_score = checkpoint['best_score']
            # patience_reached = self.args.patience
            # early_stopping = EarlyStoppingClass(patience=patience_reached, verbose=True, save_epoch_path=save_epoch_path,
            #                                     save_early_path=save_early_path,best_score=best_score)
            # print(f'STARTING FROM: epoch {epochs}')
            print(f'path: {save_epoch_path}')
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
        else:
            print('NEW TRAINING STARTED')
        # epochs = 0


        if self.args.train:
            for epoch in tqdm.tqdm(range(epochs+1,self.args.epochs),desc="Epoch"):
                # random shuffle before mini-batch training
                t_start = time.time()
                self.model.train()

                # nel training shuffle così ho sempre minibatch diverse
                num_edges_train = edge_label_index_train.size(1)
                perm_pos_train = torch.randperm(num_edges_train)
                training_positive_pd_edges = training_positive_pd_edges[:, perm_pos_train]
                perm_neg_train = torch.randperm(num_edges_train)
                training_negative_pd_edges = training_negative_pd_edges[:, perm_neg_train]

                num_edges_validation = edge_label_index_validation.size(1)
                perm_pos_validation = torch.randperm(num_edges_validation)
                validation_positive_pd_edges = validation_positive_pd_edges[:, perm_pos_validation]
                perm_neg_validation = torch.randperm(num_edges_validation)
                validation_negative_pd_edges = validation_negative_pd_edges[:, perm_neg_validation]


                train_losses = []
                val_losses = []
                for iteration in tqdm.tqdm(range(int(np.ceil(training_positive_pd_edges.size(1) / self.args.batch_size))),
                                           desc="Mini-batch"):
                    loss_train,embeddings = self.run_minibatch_transductive(self.dataset,iteration,training_positive_pd_edges,training_negative_pd_edges,all_walks=None)
                    train_losses.append(loss_train)
                    self.optimizer.zero_grad()
                    loss_train.backward()
                    self.optimizer.step()
                train_loss_final = torch.mean(torch.tensor(train_losses))

                self.model.eval()
                with torch.no_grad():
                    for iteration in tqdm.tqdm(range(int(np.ceil(validation_positive_pd_edges.size(1) / self.args.batch_size))),
                                               desc="Mini-batch"):
                        loss_validation,_ = self.run_minibatch_transductive(self.dataset,iteration,validation_positive_pd_edges,validation_negative_pd_edges,all_walks=None)

                        val_losses.append(loss_validation)
                    t_end = time.time()
                    val_loss_final = torch.mean(torch.tensor(val_losses))
                    print('Epoch {:05d} |Train loss {:.4f} | Val Loss {:.4f} | Time(s) {:.4f}'.format(
                        epoch,train_loss_final.item(), val_loss_final.item(), t_end - t_start))
                    # early stopping

                    early_stopping(val_loss_final, self.model,self.optimizer,epoch)
                    # if self.args.dataset == 'mes':
                    #     if early_stopping.best_epoch == epoch:
                    #         self.test(test_positive_pd_edges, test_negative_pd_edges, epoch,stringa,best=True)
                    #     if epoch + 1 == self.args.epochs:
                    #         self.test(test_positive_pd_edges, test_negative_pd_edges, epoch, stringa)
                        # if early_stopping.save or ( (epoch+1) % 10 == 0 and epoch != 0):
                        #     print(f'compute test for epoch: {epoch}')
                        #     self.test(test_positive_pd_edges,test_negative_pd_edges,epoch,stringa)

                    if early_stopping.early_stop:
                        # self.test(test_positive_pd_edges,test_negative_pd_edges,epoch,stringa)
                        print('Early stopping!')
                        break


        if not self.args.train:
            self.model.eval()
            with torch.no_grad():
                stringa = f'lr-{self.args.lr}_heads-{self.args.heads}_batch-{self.args.batch_size}_cores-{self.args.n_cores}_key-{self.args.n_keys_hubs}_top-{self.args.n_top_hubs}_aggrcore-{self.args.core_aggregation}_aggrkeys-{self.args.key_aggregation}_aggrtop-{self.args.top_aggregation}_allagg-{self.args.all_aggregation}'
                stringa = 'test_gen_new_'+self.args.dataset + '_' + stringa
                if args.no_metadata:
                    stringa = 'NO_METADATA_new_' + stringa

                self.test(test_positive_pd_edges, test_negative_pd_edges, 'testepoch', stringa)






if __name__ == '__main__':
    args = get_args()

    print("------arguments-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))



    # fix random seed
    # random.seed(args.random_seed)
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed_all(args.random_seed)
    # seed_everything()
    # seed_torch(args.random_seed)
    # model - ABLATION
    args = get_args()
    seed_torch(args.random_seed)
    if args.eval_lr or args.eval_all:
        args.eval_lr = True
        for lr in [0.00005,0.00001,0.0001,0.001]:
            args.lr = lr
            trainer = Trainer(args)
            trainer.run_transductive()

    args = get_args()
    seed_torch(args.random_seed)
    if args.eval_lr or args.eval_all:
        for lr in ['enriched','enriched_all','standard']:
            args.enriched = lr
            trainer = Trainer(args)
            trainer.run_transductive()

    args = get_args()
    if args.eval_heads or args.eval_all:
        args.eval_heads = True
        for lr in [1,2,4,8,16]:
            args.heads = lr
            trainer = Trainer(args)
            trainer.run_transductive()

    args = get_args()
    if args.eval_batch or args.eval_all:
        lrs = [64,256,4096]
        args.eval_batch = True
        for lr in lrs:

            args.batch_size = lr
            trainer = Trainer(args)
            trainer.run_transductive()

    args = get_args()
    if args.eval_aggregation or args.eval_all:
        args.eval_aggregation = True
        for lr in ['mh-attention','lstm']:
            args.core_aggregation = lr
            args.key_aggregation = lr
            args.top_aggregation = lr
            trainer = Trainer(args)
            trainer.run_transductive()

    args = get_args()
    if args.eval_combine_aggregation or args.eval_all:
        args.eval_combine_aggregation = True
        for lr in ['mh-attention','lstm','mean','concat']:
            args.all_aggregation = lr
            trainer = Trainer(args)
            trainer.run_transductive()

    args = get_args()
    if args.eval_neigh  or args.eval_all:
        args.eval_neigh = True
        for lr in [(12,12,10),(5,5,5),(3,3,3),(8,8,5)]:
            args.n_cores = lr[0]
            if lr[0]>5:
                args.split_cores = False
            else:
                args.split_cores = True
            args.n_keys_hubs = lr[1]
            args.n_top_hubs = lr[2]
            trainer = Trainer(args)
            trainer.run_transductive()
    else:
        # if args.dataset != 'pubmed' and not args.test:
        #     for i in ['enriched_all','enriched','standard']:
        #
        #         args = get_args()
        #         args.enriched = i
        #         trainer = Trainer(args)
        #         trainer.run_transductive()
        #         if i == 'enriched_all':
        #             args.no_metadata = True
        #             trainer = Trainer(args)
        #             trainer.run_transductive()
        # else:
            # HETGNN
        if not args.test and not args.hetgnn:
            for dataset in ['mes','pubmed']:
                args = get_args()
                args.epochs = 200
                args.batch_size=4096
                args.dataset = dataset
                args.no_metadata = False
                trainer = Trainer(args)
                trainer.run_transductive()
                args.no_metadata = True
                trainer = Trainer(args)
                trainer.run_transductive()

        elif args.hetgnn and args.train:
            #hetgnn
            for dataset in ['mes','pubmed']:
                args = get_args()
                args.epochs = 100
                args.batch = 200
                args.core_aggregation = 'lstm'
                args.key_aggregation = 'lstm'
                args.top_aggregation = 'lstm'
                args.all_aggregation = 'mh-attention'
                args.heads = 1
                args.dataset = dataset
                args.no_metadata = False
                trainer = Trainer(args)
                trainer.run_transductive()
                args.no_metadata = True
                trainer = Trainer(args)
                trainer.run_transductive()
        elif args.train:   
            for dataset in ['mes']:

                for b in [False,True]:
                    args = get_args()
                    args.dataset = dataset
                    args.hetgnn = b
                    if dataset == 'pubmed':
                        args.hetgnn = False
                    if not args.hetgnn:
                        args.epochs = 200
                        args.batch_size = 4096
                    else:
                        args.epochs = 100
                        args.epochs = 1024
                        args.core_aggregation = 'lstm'
                        args.key_aggregation = 'lstm'
                        args.top_aggregation = 'lstm'
                        args.all_aggregation = 'mh-attention'
                        args.heads = 1
                    args.no_metadata = True
                    trainer = Trainer(args)
                    trainer.run_transductive()
                    args.no_metadata = False
                    trainer = Trainer(args)
                    trainer.run_transductive()
        elif args.test:
            #for data in ['mes','pubmed']:
             #   for hg in [True,False]:
              #      for m in [True,False]:
                        args = get_args()
               #         args.dataset = data
                #        args.hetgnn = hg
                 #       args.no_metadata = m



                        if not args.hetgnn:
                            args.epochs = 200
                            args.batch_size = 4096
                            args.core_aggregation = 'mh-attention'
                            args.key_aggregation = 'mh-attention'
                            args.top_aggregation = 'mh-attention'
                            args.all_aggregation = 'concat'
                            args.beads = 8
                        else:
                            args.epochs = 100
                            args.epochs = 1024
                            args.core_aggregation = 'lstm'
                            args.key_aggregation = 'lstm'
                            args.top_aggregation = 'lstm'
                            args.all_aggregation = 'mh-attention'
                            args.heads = 1
                        trainer = Trainer(args)
                        trainer.run_transductive()
                        args.no_metadata = True
                        trainer = Trainer(args)
                        trainer.run_transductive()
