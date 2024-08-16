import torch
from args_list import get_args
import numpy as np
import random
import os

import loader_repro as loader
import sampler_repro as loader
from loader_repro import *
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
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
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
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
    def __init__(self, args,iteration,indices):
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
        precision, recall, ndcg = precision / c, recall / c, ndcg / c

        reranking_line = 'GOAL precision = {}'.format(precision) + ' recall = {}'.format(
            recall) + ' ndcg = {}'.format(ndcg) + '\n'
        print(reranking_line)
        return mapped_results

    def test(self, test_positive_pd_edges, test_negative_pd_edges, epoch, stringa):
        with torch.no_grad():

            output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/bootstrapped/{self.args.bootstrap}/{self.iteration}_{epoch}_{stringa}.txt"
            # if best:
            #     output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/bootstrapped/{self.args.bootstrap}/BEST_{self.iteration}_{epoch}_{stringa}.txt"
            # if epoch + 1 == self.args.epochs:
            #     output_file_path = f"model/checkpoint/transductive/results/{self.args.dataset}/bootstrapped/{self.args.bootstrap}/LAST_{self.iteration}_{epoch}_{stringa}.txt"

            f = open(output_file_path, 'a')
            f.write(stringa + '\n')
            self.model.eval()
            auc_tot = 0
            ap_tot = 0
            rec_tot = 0
            prec_tot = 0
            ndcg_tot = 0

            for i in range(10):
                print('eval round: ', str(i))
                # LINK PREDICTION
                y_true_test = np.array([1] * test_positive_pd_edges.size(1) + [0] * test_negative_pd_edges.size(1))
                # walks = self.get_walks()
                # all_walks = {seed: [] for seed in self.walker.G.nodes if self.walker.is_publication(seed) or self.walker.is_dataset(seed)}
                #
                # for walk in walks:
                #     all_walks[walk[0]].append(walk)
                # all_walks = {k:v for k,v in all_walks.items() if len(v) > 0}
                # all_walks = walker.create_random_walks(seeds_in=all_seeds)
                loss, final_embeddings, pos_source, pos_target, neg_source, neg_target, all_seeds = self.run_minibatch_transductive(
                    self.dataset, 0,
                    test_positive_pd_edges, test_negative_pd_edges,
                    test=True, all_walks=None)

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
                # walks = self.get_walks()
                # self.walker.set_seeds(all_seeds)
                # all_walks = {seed: [] for seed in self.walker.G.nodes if self.walker.is_publication(seed) or self.walker.is_dataset(seed)}
                # for walk in walks:
                #     all_walks[walk[0]].append(walk)
                # all_walks = {k:v for k,v in all_walks.items() if len(v) > 0}
                # all_walks = [v for k, v in all_walks.items() if k in all_seeds]
                # all_walks = [inner for outer in all_walks for inner in outer]
                self.walker.set_seeds(all_seeds)
                all_walks = self.walker.create_random_walks(seeds_in=all_seeds)
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

                y_test_predicted_labels = {source: [] for source in sources}
                y_test_predicted_labels_norerank = {source: [] for source in sources}
                # mapped_results = self.trivial_baselines()
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
                    # print(true)
                    # print(pred)

                    # for p in pred:
                    #     run.append(f'{self.dataset["publication"].rev_mapping[source]}\tQ0\t{self.dataset["dataset"].rev_mapping[p]}\t{pred.index(p)+1}\t{pred.index(p)+1}\tmyrun\n')
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
                no_rer_precision, no_rer_recall, no_rer_ndcg = no_rer_precision / len(sources), no_rer_recall / len(
                    sources), no_rer_ndcg / len(sources)
                prec_tot += no_rer_precision
                rec_tot += no_rer_recall
                ndcg_tot += no_rer_ndcg
                auc_tot += auc
                ap_tot += ap

            prec_tot = prec_tot / 10
            rec_tot = rec_tot / 10
            ndcg_tot = ndcg_tot / 10
            auc_tot = auc_tot / 10
            ap_tot = ap_tot / 10

            reranking_line = 'AP ={}'.format(ap_tot) + 'AUC ={}'.format(
                auc_tot) + '\n' + 'STANDARD precision = {}'.format(prec_tot) + ' recall = {}'.format(
                rec_tot) + ' ndcg = {}'.format(ndcg_tot) + '\n'
            print(reranking_line)
            f.write(reranking_line)

            f.close()
            return ap_tot,auc_tot,prec_tot,rec_tot,ndcg_tot

    def run_minibatch_transductive(self, dataset, iteration, positive_edges, negative_edges, test=False,
                                   all_walks=None):

        # seleziono gli indici dell'edge index che mi interessano. Divido per due la batch: voglio ugual numero di archi positivi e negativi
        # i nodi satanno al più il doppio della batchsize perchè ogni arco ha due nodi
        if not test:
            batch_positive = positive_edges[:, iteration * self.args.batch_size: (iteration + 1) * self.args.batch_size]
            batch_negative = negative_edges[:, iteration * self.args.batch_size: (iteration + 1) * self.args.batch_size]

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


        pos_source_indices = [all_seeds.index(j) for i, j in enumerate(mapped_sources)]
        pos_target_indices = [all_seeds.index(j) for i, j in enumerate(mapped_targets)]
        neg_source_indices = [all_seeds.index(j) for i, j in enumerate(mapped_neg_sources)]
        neg_target_indices = [all_seeds.index(j) for i, j in enumerate(mapped_neg_targets)]

        # # # seed_vectors: dim = 3: seed, mapped seed, vector
        # # # remaining: dim = 5: seed, id, score, mapped_id, vector
        if self.args.verbose:
            print('start model')

        selected_seeds_walks, selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top = self.walker.select_walks_mp(
            all_walks)
        seed_vectors, seed_vectors_net, net_cores, cores, net_keys, keys, hubs, _, _, _, _, _, _, _, _ = self.walker.get_neighbours_vector(
            selected_seeds_cores, selected_seeds_hubs_key, selected_seeds_hubs_top)
        final_embeddings = self.model(seed_vectors, seed_vectors_net, net_cores, cores, net_keys, keys, hubs,
                                      core_agg=self.args.core_aggregation, key_agg=self.args.key_aggregation,
                                      top_agg=self.args.top_aggregation, all_agg=self.args.all_aggregation)
        loss = self.model.cross_entropy_loss(final_embeddings, pos_source_indices, pos_target_indices,
                                             neg_source_indices, neg_target_indices)
        if test:
            return loss, final_embeddings, pos_source_indices, pos_target_indices, neg_source_indices, neg_target_indices, all_seeds
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

        edge_label_index_train = self.dataset['publication', 'cites', 'dataset'].edge_label_index_train
        edge_label_index_validation = self.dataset['publication', 'cites', 'dataset'].edge_label_index_validation
        edge_label_index_test = self.dataset['publication', 'cites', 'dataset'].edge_label_index_test

        print('Generating negative edges')
        training_positive_pd_edges = edge_label_index_train
        validation_positive_pd_edges = edge_label_index_validation
        test_positive_pd_edges = edge_label_index_test

        # if self.args.train:
        # training_negative_pd_edges = loader.cosine_based_negative_samples(edge_index_train,training_positive_pd_edges,source_vectors,target_vectors,similar=False)
        training_negative_pd_edges = self.dataset['publication', 'cites', 'dataset'].negative_edge_label_index_train
        validation_negative_pd_edges = self.dataset[
            'publication', 'cites', 'dataset'].negative_edge_label_index_validation
        test_negative_pd_edges = self.dataset['publication', 'cites', 'dataset'].negative_edge_label_index_test
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
        stringa = 'test_gen_new_' + self.args.dataset + '_' + stringa

        save_epoch_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/bootstrapped/{self.args.bootstrap}/{self.iteration}_last_checkpoint_{stringa}_last_epoch.pt'
        save_early_path = f'./model/checkpoint/transductive/models/{self.args.dataset}/bootstrapped/{self.args.bootstrap}/{self.iteration}_best_checkpoint_{stringa}.pt'

        early_stopping = EarlyStoppingClass(patience=args.patience, verbose=True, save_epoch_path=save_epoch_path,
                                            save_early_path=save_early_path)
        epochs = -1
        print(os.path.exists(save_epoch_path))
        if os.path.exists(save_epoch_path) and not self.args.restart:
            checkpoint = torch.load(save_epoch_path)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epochs = checkpoint['epoch']
                best_score = checkpoint['best_score']
                patience_reached = self.args.patience
                early_stopping = EarlyStoppingClass(patience=patience_reached, verbose=True,
                                                    save_epoch_path=save_epoch_path,
                                                    save_early_path=save_early_path, best_score=best_score)
            except:
                self.model.load_state_dict(checkpoint)
            print(f'STARTING FROM: epoch {epochs}')
        else:
            print('NEW TRAINING STARTED')
        # epochs = 0

        if self.args.train:
            for epoch in tqdm.tqdm(range(epochs + 1, self.args.epochs), desc="Epoch"):
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
                for iteration in tqdm.tqdm(
                        range(int(np.ceil(training_positive_pd_edges.size(1) / self.args.batch_size))),
                        desc="Mini-batch"):
                    loss_train, embeddings = self.run_minibatch_transductive(self.dataset, iteration,
                                                                             training_positive_pd_edges,
                                                                             training_negative_pd_edges, all_walks=None)
                    train_losses.append(loss_train)
                    self.optimizer.zero_grad()
                    loss_train.backward()
                    self.optimizer.step()
                train_loss_final = torch.mean(torch.tensor(train_losses))

                self.model.eval()
                with torch.no_grad():
                    for iteration in tqdm.tqdm(
                            range(int(np.ceil(validation_positive_pd_edges.size(1) / self.args.batch_size))),
                            desc="Mini-batch"):
                        loss_validation, _ = self.run_minibatch_transductive(self.dataset, iteration,
                                                                             validation_positive_pd_edges,
                                                                             validation_negative_pd_edges,
                                                                             all_walks=None)

                        val_losses.append(loss_validation)
                    t_end = time.time()
                    val_loss_final = torch.mean(torch.tensor(val_losses))
                    print('Epoch {:05d} |Train loss {:.4f} | Val Loss {:.4f} | Time(s) {:.4f}'.format(
                        epoch, train_loss_final.item(), val_loss_final.item(), t_end - t_start))
                    # early stopping

                    early_stopping(val_loss_final, self.model, self.optimizer, epoch)
                    # if early_stopping.best_epoch == epoch:
                    #     self.test(test_positive_pd_edges, test_negative_pd_edges, epoch,stringa,best=True)
                    # if epoch + 1 == self.args.epochs:
                    #     self.test(test_positive_pd_edges, test_negative_pd_edges, epoch, stringa)

        if not self.args.train:
            self.model.eval()
            with torch.no_grad():
                stringa = f'lr-{self.args.lr}_heads-{self.args.heads}_batch-{self.args.batch_size}_cores-{self.args.n_cores}_key-{self.args.n_keys_hubs}_top-{self.args.n_top_hubs}_aggrcore-{self.args.core_aggregation}_aggrkeys-{self.args.key_aggregation}_aggrtop-{self.args.top_aggregation}_allagg-{self.args.all_aggregation}'
                stringa = 'test_gen_new_' + self.args.dataset + '_' + stringa
                ap_tot,auc_tot,prec_tot,rec_tot,ndcg_tot = self.test(test_positive_pd_edges, test_negative_pd_edges, 'testepoch', stringa)
                return ap_tot,auc_tot,prec_tot,rec_tot,ndcg_tot

if __name__ == '__main__':
    args = get_args()

    print("------arguments-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))


    args = get_args()
    seed_torch(args.random_seed)
    for boot in [False]:
        args = get_args()

        args.hetgnn = boot

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

        indices_dict = {}
        if args.train and args.dataset == 'mes':
            for b in [25,50,75]:
                args.bootstrap = b
                outpath = f'model/checkpoint/transductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices_{args.batch_size}.json'
                if args.hetgnn:
                    outpath = f'model/checkpoint/transductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices_{args.batch_size}_hetgnn.json'

                for i in range(0,10):
                    print(f'ITERATION: {i}')
                    if i > 0 or os.path.exists(outpath):
                        f = open(outpath,'r')
                        indices_dict = json.load(f)
    
                    number_perm = int((int(args.bootstrap)/100)*2949)
                    indices = random.sample(range(2949), number_perm)
                    indices_dict[f'iter_{str(i)}'] = indices
                    f = open(outpath, 'w')
                    json.dump(indices_dict,f)
                    print(len(indices))
                    trainer = Trainer(args,i,indices)
                    trainer.run_transductive()

        elif args.dataset == 'pubmed':
            outpath = f'model/checkpoint/transductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices_{args.batch_size}.json'
            if args.hetgnn:
                outpath = f'model/checkpoint/transductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices_{args.batch_size}_hetgnn.json'

            for i in range(int(args.iteration), int(args.iteration + 1)):
                print(f'ITERATION: {i}')

                if i > 0 or os.path.exists(outpath):
                    f = open(outpath, 'r')
                    indices_dict = json.load(f)
                    f.close()
                number_perm = int((int(args.bootstrap) / 100) * data_num)
                indices = random.sample(range(data_num), number_perm)

 
                indices_dict[f'iter_{str(i)}'] = indices
                g = open(outpath, 'w')
                json.dump(indices_dict, g)
                g.close()
                trner(args, i, indices)
                trainer.run_inductive()
                g.close()


        if args.test:
            for b in [25,50,75]:
                # args = get_args()
                args.hetgnn = boot
                args.bootstrap = b
                outpath = f'model/checkpoint/transductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices_{args.batch_size}.json'
                if args.hetgnn:
                    outpath = f'model/checkpoint/transductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/indices_{args.batch_size}_hetgnn.json'

                ap, auc, prec, rec, ndcg = [],[],[],[],[]
                for i in range(0, 10):
                    f = open(outpath,
                             'r')
                    indices_dict = json.load(f)
                    indices = indices_dict[f'iter_{str(i)}']
                    print(f'len indices {len(indices)}')
                    trainer = Trainer(args, i, indices)
                    ap_tot,auc_tot,prec_tot,rec_tot,ndcg_tot = trainer.run_transductive()
                    ap.append(ap_tot)
                    auc.append(auc_tot)
                    prec.append(prec_tot)
                    rec.append(rec_tot)
                    ndcg.append(ndcg_tot)
                    b = sum(ap) / 10

                output_file_path = f"model/checkpoint/transductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/recap_{args.batch_size}_{args.epochs}.txt"
                if args.hetgnn:
                    output_file_path = f"model/checkpoint/transductive/results/{args.dataset}/bootstrapped/{args.bootstrap}/recap_{args.batch_size}_{args.epochs}_hetgnn.txt"
    
                f = open(output_file_path, 'w')
                f.write(f"AP {sum(ap) / 10} AUC {sum(auc) / 10}\n P {sum(prec)/10} R {sum(rec) / 10} NDCG {sum(ndcg) / 10}\n")
                f.write(f"AP {np.std(ap) / np.sqrt(10)} AUC {np.std(auc) / np.sqrt(10)}\n P {np.std(prec)/np.sqrt(10)} R {np.std(rec) / np.sqrt(10)} NDCG {np.std(ndcg) / np.sqrt(10)}")
                f.close()
