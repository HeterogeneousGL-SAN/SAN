import os.path as osp
import pandas as pd
import torch_geometric.utils.convert
from sentence_transformers import SentenceTransformer
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.data import Data, InMemoryDataset
import pickle
import argparse
from gensim.models import KeyedVectors
import multiprocessing as mp
import json
import random
import torch
import numpy as np
import utils
from args_list import get_args
import os
args = get_args()
EMBEDDING_MODEL = f'model/data/{args.dataset}/embeddings/node2vec'
seed=42
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

def load_node_csv(path, index_col, encoders=None, **kwargs):

    df = pd.read_csv(path, index_col=index_col, **kwargs)
    print(path,df.shape[0])
    if 'content' in list(df.columns):
        df['content'] = df['content'].astype(str)
    if 'name' in list(df.columns):
        df['name'] = df['name'].astype(str)
    if 'fullname' in list(df.columns):
        df['fullname'] = df['fullname'].astype(str)
    if 'description' in list(df.columns):
        df['description'] = df['description'].astype(str)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    model = KeyedVectors.load_word2vec_format(f"{EMBEDDING_MODEL}/node2vec_model.bin")

    embeddings = []
    zeros = 0
    for k,v in mapping.items():
        if k in model.vocab:
            embeddings.append(model[str(k)])
        else:
            embeddings.append(torch.zeros(128).numpy())
            zeros +=1

    net_x = torch.tensor(embeddings)
    return x,net_x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)
    print(path,df.shape[0])

    def switch_values(row):
        if row['source'].startswith('d_') and row['target'].startswith('p_'):
            return pd.Series({'source': row['target'], 'target': row['source']})
        else:
            return row

    df = df.apply(switch_values, axis=1)
    df = df.drop_duplicates(keep='first')

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class ContentEncoder:
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()

class KeywordEncoder:
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='whaleloops/phrase-bert', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()



class ScholarlyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if 'transductive' in self.root:
            return ['graph_transductive_kcore.pt']

        elif 'inductive_full' in self.root:
            return ['graph_inductive_full_0.pt']

        elif 'inductive_light' in self.root:
            return ['graph_inductive_light_0.pt']



    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        dataset = self.root.split('/')[1]
        print(f'DATASET: {dataset}')
        if 'transductive' in self.root:
            data_pre = self.create_transductive_graph()
            data_list = [data_pre]
        elif 'inductive_light' in self.root:
            data_train = self.create_inductive_graph()
            # data_vali = self.create_inductive_graph()
            # data_test = self.create_inductive_graph()
            data_list = [data_train]
        elif 'inductive_full' in self.root:
            data_train = self.create_inductive_graph()
            # data_vali = self.create_inductive_graph()
            # data_test = self.create_inductive_graph()
            data_list = [data_train]

        print(self.root)



        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def create_transductive_graph(self):
        publication_path = self.root + '/publications.csv'
        dataset_path = self.root + '/datasets.csv'
        authors_path = self.root + '/authors.csv'
        topic_path = self.root + '/topics_keywords_2.csv'
        entity_path = self.root + '/entities.csv'

        publication_x,publication_net_x, publication_mapping = load_node_csv(publication_path, index_col='id', encoders={'content': ContentEncoder()})
        print(publication_net_x.shape)
        dataset_x,dataset_net_x, dataset_mapping = load_node_csv(dataset_path, index_col='id', encoders={'content': ContentEncoder()})
        authors_x,authors_net_x, author_mapping = load_node_csv(authors_path, index_col='id',encoders={'fullname': KeywordEncoder()})
        topic_x,topic_net_x, topic_mapping = load_node_csv(topic_path, index_col='id',encoders={'description': KeywordEncoder()    })
        entity_x,entity_net_x, entity_mapping = load_node_csv(entity_path, index_col='id',encoders={'name': KeywordEncoder()    })

        if 'mes' not in self.root:
            keyword_path = self.root + '/keywords.csv'
            orgs_path = self.root + '/organizations.csv'
            venue_path = self.root + '/venues.csv'
            orgs_x,orgs_net_x, orgs_mapping = load_node_csv(orgs_path, index_col='id',encoders={'name': KeywordEncoder()})
            venue_x,venue_net_x, venue_mapping = load_node_csv(venue_path, index_col='id',encoders={'name': KeywordEncoder()})
            keyword_x,keyword_net_x, keyword_mapping = load_node_csv(keyword_path, index_col='id', encoders={'name': KeywordEncoder()})

        pd_edges_path_train = self.root + '/pubdataedges_train_kcore_1.csv'
        pd_edges_path_vali = self.root + '/pubdataedges_validation_kcore_1.csv'
        pd_edges_path_test = self.root + '/pubdataedges_test_kcore_1.csv'
        pp_edges_path = self.root + '/pubpubedges.csv'
        dd_edges_path = self.root + '/datadataedges.csv'
        pa_edges_path = self.root + '/pubauthedges.csv'
        da_edges_path = self.root + '/dataauthedges.csv'
        pe_edges_path = self.root + '/pubentedges.csv'
        de_edges_path = self.root + '/dataentedges.csv'
        pt_edges_path = self.root + '/pubtopicedges_keywords_2.csv'
        dt_edges_path = self.root + '/datatopicedges_keywords_2.csv'

        if 'mes' not in self.root:
            pv_edges_path = self.root + '/pubvenuesedges.csv'
            po_edges_path = self.root + '/puborgedges.csv'
            do_edges_path = self.root + '/dataorgedges.csv'
            pk_edges_path = self.root + '/pubkeyedges.csv'
            dk_edges_path = self.root + '/datakeyedges.csv'

        pd_edge_index_train, pd_edge_label = load_edge_csv(
            pd_edges_path_train,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )
        pd_edge_index_validation, pd_edge_label = load_edge_csv(
            pd_edges_path_vali,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )
        pd_edge_index_test, pd_edge_label = load_edge_csv(
            pd_edges_path_test,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )

        pp_edge_index, pp_edge_label = load_edge_csv(
            pp_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=publication_mapping,
        )

        dd_edge_index, dd_edge_label = load_edge_csv(
            dd_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )

        pa_edge_index, pa_edge_label = load_edge_csv(
            pa_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=author_mapping,
        )
        da_edge_index, da_edge_label = load_edge_csv(
            da_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=author_mapping,
        )
        de_edge_index, de_edge_label = load_edge_csv(
            de_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=entity_mapping,
        )
        pe_edge_index, pe_edge_label = load_edge_csv(
            pe_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=entity_mapping,
        )
        pt_edge_index, pt_edge_label = load_edge_csv(
            pt_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=topic_mapping,
        )
        dt_edge_index, dt_edge_label = load_edge_csv(
            dt_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=topic_mapping,
        )
        if 'mes' not in self.root:
            pv_edge_index, pv_edge_label = load_edge_csv(
                pv_edges_path,
                src_index_col='source',
                src_mapping=publication_mapping,
                dst_index_col='target',
                dst_mapping=venue_mapping,
            )
            po_edge_index, po_edge_label = load_edge_csv(
                po_edges_path,
                src_index_col='source',
                src_mapping=publication_mapping,
                dst_index_col='target',
                dst_mapping=orgs_mapping,
            )
            do_edge_index, do_edge_label = load_edge_csv(
                do_edges_path,
                src_index_col='source',
                src_mapping=dataset_mapping,
                dst_index_col='target',
                dst_mapping=orgs_mapping,
            )
            pk_edge_index, pk_edge_label = load_edge_csv(
                pk_edges_path,
                src_index_col='source',
                src_mapping=publication_mapping,
                dst_index_col='target',
                dst_mapping=keyword_mapping,
            )
            dk_edge_index, dk_edge_label = load_edge_csv(
                dk_edges_path,
                src_index_col='source',
                src_mapping=dataset_mapping,
                dst_index_col='target',
                dst_mapping=keyword_mapping,
            )

        data = HeteroData()
        data['publication'].x = publication_x
        data['publication'].net_x = publication_net_x
        data['publication'].mapping = publication_mapping
        data['publication'].num_nodes = len(publication_mapping)  # Users do not have any features.
        data['publication'].rev_mapping = {v:k for k,v in publication_mapping.items()}

        data['dataset'].num_nodes = len(dataset_mapping)  # Users do not have any features.
        data['dataset'].x = dataset_x
        data['dataset'].net_x = dataset_net_x
        data['dataset'].mapping = dataset_mapping
        data['dataset'].rev_mapping = {v:k for k,v in dataset_mapping.items()}


        data['author'].num_nodes = len(author_mapping)  # Users do not have any features.
        data['author'].x = authors_x
        data['author'].net_x = authors_net_x
        data['author'].mapping = author_mapping
        data['author'].rev_mapping = {v:k for k,v in author_mapping.items()}


        data['topic'].num_nodes = len(topic_mapping)  # Users do not have any features.
        data['topic'].x = topic_x
        data['topic'].net_x = topic_net_x
        data['topic'].mapping = topic_mapping
        data['topic'].rev_mapping = {v:k for k,v in topic_mapping.items()}


        data['entity'].num_nodes = len(entity_mapping)  # Users do not have any features.
        data['entity'].x = entity_x
        data['entity'].net_x = entity_net_x
        data['entity'].mapping = entity_mapping
        data['entity'].rev_mapping = {v:k for k,v in entity_mapping.items()}


        if 'mes' not in self.root:
            data['venue'].num_nodes = len(venue_mapping)  # Users do not have any features.
            data['venue'].x = venue_x
            data['venue'].net_x = venue_net_x
            data['venue'].mapping = venue_mapping
            data['venue'].rev_mapping = {v: k for k, v in entity_mapping.items()}

            data['keyword'].num_nodes = len(keyword_mapping)  # Users do not have any features.
            data['keyword'].x = keyword_x
            data['keyword'].net_x = keyword_net_x
            data['keyword'].mapping = keyword_mapping
            data['keyword'].rev_mapping = {v: k for k, v in keyword_mapping.items()}

            data['organization'].num_nodes = len(orgs_mapping)  # Users do not have any features.
            data['organization'].x = orgs_x
            data['organization'].net_x = orgs_net_x
            data['organization'].mapping = orgs_mapping
            data['organization'].rev_mapping = {v:k for k,v in orgs_mapping.items()}


        # leave part of edges for message passing
        data['publication', 'cites', 'dataset'].edge_index_train = pd_edge_index_train
        data['publication', 'cites', 'dataset'].edge_label_index_train = pd_edge_index_train
        cosine_label_ind = torch.cat([pd_edge_index_train,pd_edge_index_validation,pd_edge_index_test],dim=1)
        data['publication', 'cites', 'dataset'].negative_edge_label_index_train = cosine_based_negative_samples(cosine_label_ind,pd_edge_index_train,data['publication'].x,data['dataset'].x)

        data['publication', 'cites', 'dataset'].edge_index_validation = pd_edge_index_train
        data['publication', 'cites', 'dataset'].edge_label_index_validation = pd_edge_index_validation
        data['publication', 'cites', 'dataset'].negative_edge_label_index_validation = cosine_based_negative_samples(cosine_label_ind,pd_edge_index_validation,data['publication'].x,data['dataset'].x)


        data['publication', 'cites', 'dataset'].edge_index_test = torch.cat([pd_edge_index_train,pd_edge_index_validation],dim=1)
        data['publication', 'cites', 'dataset'].edge_label_index_test = pd_edge_index_test
        data['publication', 'cites', 'dataset'].negative_edge_label_index_test = cosine_based_negative_samples(torch.cat([cosine_label_ind,pd_edge_index_validation],dim=1),pd_edge_index_test,data['publication'].x,data['dataset'].x)


        data['publication', 'cites', 'publication'].edge_index = pp_edge_index
        data['publication', 'cites', 'publication'].edge_label = pp_edge_label
        data['publication', 'hasauthor', 'author'].edge_index = pa_edge_index
        data['publication', 'hasauthor', 'author'].edge_label = pa_edge_label
        data['dataset', 'hasauthor', 'author'].edge_index = da_edge_index
        data['dataset', 'hasauthor', 'author'].edge_label = da_edge_label
        data['dataset', 'cites', 'dataset'].edge_index = dd_edge_index
        data['dataset', 'cites', 'dataset'].edge_label = dd_edge_label
        data['publication', 'hasentity', 'entity'].edge_index = pe_edge_index
        data['publication', 'hasentity', 'entity'].edge_label = pe_edge_label
        data['dataset', 'hasentity', 'entity'].edge_index = de_edge_index
        data['dataset', 'hasentity', 'entity'].edge_label = de_edge_label
        data['publication', 'hastopic', 'topic'].edge_index = pt_edge_index
        data['publication', 'hastopic', 'topic'].edge_label = pt_edge_label
        data['dataset', 'hastopic', 'topic'].edge_index = dt_edge_index
        data['dataset', 'hastopic', 'topic'].edge_label = dt_edge_label

        if 'mes' not in self.root:
            data['publication', 'hasvenue', 'venue'].edge_index = pv_edge_index
            data['publication', 'hasvenue', 'venue'].edge_label = pv_edge_label
            data['publication', 'haskeyword', 'keyword'].edge_index = pk_edge_index
            data['publication', 'haskeyword', 'keyword'].edge_label = pk_edge_label
            data['dataset', 'haskeyword', 'keyword'].edge_index = dk_edge_index
            data['dataset', 'haskeyword', 'keyword'].edge_label = dk_edge_label
            data['publication', 'hasorganization', 'organization'].edge_index = po_edge_index
            data['publication', 'hasorganization', 'organization'].edge_label = po_edge_label
            data['dataset', 'hasorganization', 'organization'].edge_index = do_edge_index
            data['dataset', 'hasorganization', 'organization'].edge_label = do_edge_label

        print(data)

        return data

    def create_inductive_graph(self):
        dataset = self.root.split('/')[1]
        r = f'datasets/{dataset}/split_transductive/train'
        data = HeteroData()
        publication_path = r + '/publications.csv'
        dataset_path = r + '/datasets.csv'
        authors_path = r + '/authors.csv'
        topic_path = r + '/topics_keywords_2.csv'
        entity_path = r + '/entities.csv'

        publication_x,publication_net_x, publication_mapping = load_node_csv(publication_path, index_col='id', encoders={'content': ContentEncoder()})
        dataset_x,dataset_net_x, dataset_mapping = load_node_csv(dataset_path, index_col='id', encoders={'content': ContentEncoder()})
        authors_x,authors_net_x, author_mapping = load_node_csv(authors_path, index_col='id',encoders={'fullname': KeywordEncoder()})
        topic_x,topic_net_x, topic_mapping = load_node_csv(topic_path, index_col='id',encoders={'description': KeywordEncoder()    })
        entity_x,entity_net_x, entity_mapping = load_node_csv(entity_path, index_col='id',encoders={'name': KeywordEncoder()    })

        if 'mes' not in self.root:
            keyword_path = r + '/keywords.csv'
            orgs_path = r + '/organizations.csv'
            venue_path = r + '/venues.csv'
            orgs_x,orgs_net_x, orgs_mapping = load_node_csv(orgs_path, index_col='id',encoders={'name': KeywordEncoder()})
            venue_x,venue_net_x, venue_mapping = load_node_csv(venue_path, index_col='id',encoders={'name': KeywordEncoder()})
            keyword_x,keyword_net_x, keyword_mapping = load_node_csv(keyword_path, index_col='id', encoders={'name': KeywordEncoder()})

        print(self.root)
        pd_edges_path = self.root + '/pubdataedges.csv'
        pp_edges_path = self.root + '/pubpubedges.csv'
        dd_edges_path = self.root + '/datadataedges.csv'
        pa_edges_path = self.root + '/pubauthedges.csv'
        da_edges_path = self.root + '/dataauthedges.csv'
        pe_edges_path = self.root + '/pubentedges.csv'
        de_edges_path = self.root + '/dataentedges.csv'
        pt_edges_path = self.root + '/pubtopicedges_keywords_2.csv'
        dt_edges_path = self.root + '/datatopicedges_keywords_2.csv'

        if 'mes' not in self.root:
            pv_edges_path = self.root + '/pubvenuesedges.csv'
            po_edges_path = self.root + '/puborgedges.csv'
            do_edges_path = self.root + '/dataorgedges.csv'
            pk_edges_path = self.root + '/pubkeyedges.csv'
            dk_edges_path = self.root + '/datakeyedges.csv'

        pd_edge_index, pd_edge_label = load_edge_csv(
            pd_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )

        pd_edge_index_mp = pd_edge_index[:, :int( pd_edge_index.size(1) / 2)]
        pd_edge_index_label = pd_edge_index[:, int( pd_edge_index.size(1) / 2):]
        if 'train' in self.root:
            pd_edge_index_mp = pd_edge_index[:,:int(2*pd_edge_index.size(1)/3)]
            pd_edge_index_label = pd_edge_index[:,int(2*pd_edge_index.size(1)/3):]


        pp_edge_index, pp_edge_label = load_edge_csv(
            pp_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=publication_mapping,
        )

        dd_edge_index, dd_edge_label = load_edge_csv(
            dd_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )

        pa_edge_index, pa_edge_label = load_edge_csv(
            pa_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=author_mapping,
        )
        da_edge_index, da_edge_label = load_edge_csv(
            da_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=author_mapping,
        )
        de_edge_index, de_edge_label = load_edge_csv(
            de_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=entity_mapping,
        )
        pe_edge_index, pe_edge_label = load_edge_csv(
            pe_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=entity_mapping,
        )
        pt_edge_index, pt_edge_label = load_edge_csv(
            pt_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=topic_mapping,
        )
        dt_edge_index, dt_edge_label = load_edge_csv(
            dt_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=topic_mapping,
        )
        if 'mes' not in self.root:
            pv_edge_index, pv_edge_label = load_edge_csv(
                pv_edges_path,
                src_index_col='source',
                src_mapping=publication_mapping,
                dst_index_col='target',
                dst_mapping=venue_mapping,
            )
            po_edge_index, po_edge_label = load_edge_csv(
                po_edges_path,
                src_index_col='source',
                src_mapping=publication_mapping,
                dst_index_col='target',
                dst_mapping=orgs_mapping,
            )
            do_edge_index, do_edge_label = load_edge_csv(
                do_edges_path,
                src_index_col='source',
                src_mapping=dataset_mapping,
                dst_index_col='target',
                dst_mapping=orgs_mapping,
            )
            pk_edge_index, pk_edge_label = load_edge_csv(
                pk_edges_path,
                src_index_col='source',
                src_mapping=publication_mapping,
                dst_index_col='target',
                dst_mapping=keyword_mapping,
            )
            dk_edge_index, dk_edge_label = load_edge_csv(
                dk_edges_path,
                src_index_col='source',
                src_mapping=dataset_mapping,
                dst_index_col='target',
                dst_mapping=keyword_mapping,
            )

        data['publication'].x = publication_x
        data['publication'].net_x = publication_net_x
        data['publication'].mapping = publication_mapping
        data['publication'].num_nodes = len(publication_mapping)  # Users do not have any features.

        data['dataset'].num_nodes = len(dataset_mapping)  # Users do not have any features.
        data['dataset'].x = dataset_x
        data['dataset'].net_x = dataset_net_x
        data['dataset'].mapping = dataset_mapping

        data['author'].num_nodes = len(author_mapping)  # Users do not have any features.
        data['author'].x = authors_x
        data['author'].net_x = authors_net_x
        data['author'].mapping = author_mapping

        data['topic'].num_nodes = len(topic_mapping)  # Users do not have any features.
        data['topic'].x = topic_x
        data['topic'].net_x = topic_net_x
        data['topic'].mapping = topic_mapping

        data['entity'].num_nodes = len(entity_mapping)  # Users do not have any features.
        data['entity'].x = entity_x
        data['entity'].net_x = entity_net_x
        data['entity'].mapping = entity_mapping

        if 'mes' not in self.root:
            data['venue'].num_nodes = len(venue_mapping)  # Users do not have any features.
            data['venue'].x = venue_x
            data['venue'].net_x = venue_net_x
            data['venue'].mapping = venue_mapping

            data['keyword'].num_nodes = len(keyword_mapping)  # Users do not have any features.
            data['keyword'].x = keyword_x
            data['keyword'].net_x = keyword_net_x
            data['keyword'].mapping = keyword_mapping

            data['organization'].num_nodes = len(orgs_mapping)  # Users do not have any features.
            data['organization'].x = orgs_x
            data['organization'].net_x = orgs_net_x
            data['organization'].mapping = organization_mapping

        data['publication', 'cites', 'dataset'].edge_index = pd_edge_index_mp
        data['publication', 'cites', 'dataset'].edge_label_index = pd_edge_index_label
        data['publication', 'cites', 'publication'].edge_index = pp_edge_index
        data['publication', 'cites', 'publication'].edge_label = pp_edge_label
        data['publication', 'hasauthor', 'author'].edge_index = pa_edge_index
        data['publication', 'hasauthor', 'author'].edge_label = pa_edge_label
        data['dataset', 'hasauthor', 'author'].edge_index = da_edge_index
        data['dataset', 'hasauthor', 'author'].edge_label = da_edge_label
        data['dataset', 'cites', 'dataset'].edge_index = dd_edge_index
        data['dataset', 'cites', 'dataset'].edge_label = dd_edge_label
        data['publication', 'hasentity', 'entity'].edge_index = pe_edge_index
        data['publication', 'hasentity', 'entity'].edge_label = pe_edge_label
        data['dataset', 'hasentity', 'entity'].edge_index = de_edge_index
        data['dataset', 'hasentity', 'entity'].edge_label = de_edge_label
        data['publication', 'hastopic', 'topic'].edge_index = pt_edge_index
        data['publication', 'hastopic', 'topic'].edge_label = pt_edge_label
        data['dataset', 'hastopic', 'topic'].edge_index = dt_edge_index
        data['dataset', 'hastopic', 'topic'].edge_label = dt_edge_label

        if 'mes' not in self.root:
            data['publication', 'hasvenue', 'venue'].edge_index = pv_edge_index
            data['publication', 'hasvenue', 'venue'].edge_label = pv_edge_label
            data['publication', 'haskeyword', 'keyword'].edge_index = pk_edge_index
            data['publication', 'haskeyword', 'keyword'].edge_label = pk_edge_label
            data['dataset', 'haskeyword', 'keyword'].edge_index = dk_edge_index
            data['dataset', 'haskeyword', 'keyword'].edge_label = dk_edge_label
            data['publication', 'hasorganization', 'organization'].edge_index = po_edge_index
            data['publication', 'hasorganization', 'organization'].edge_label = po_edge_label
            data['dataset', 'hasorganization', 'organization'].edge_index = do_edge_index
            data['dataset', 'hasorganization', 'organization'].edge_label = do_edge_label


        return data


def cosine_based_negative_samples(edge_index,edge_label_index,source_vectors,target_vectors,similar=False):
    edges_all = torch.cat([edge_index,edge_label_index],dim=1)
    edges_t = edges_all.t().tolist()
    edges_t = list(set(tuple(e) for e in edges_t))

    # print(source_vectors[0])
    sources = list(set(edges_all[0].tolist()))
    targets = list(set(edges_all[1].tolist()))
    negative_edges = []
    total_negative_samples = edge_label_index.size(1)
    # print(len(sources))
    # print(total_negative_samples)
    # if len(sources) < total_negative_samples:
    #     sources = sources + sources
    # random_sources = random.sample(sources,total_negative_samples)
    # print(f'total random negative edges {total_negative_samples}')
    while len(negative_edges) < total_negative_samples:
        s = random.sample(sources,1)[0]
        random.shuffle(targets)
        targets_selected = random.sample(targets, 100)
        targets_cosine = []
        for j,t in enumerate(targets_selected):
            # print('working on target: ', t, j)
            if tuple([s, t]) not in edges_t:
                cosine = cosine_similarity(source_vectors[s],target_vectors[t])
                targets_cosine.append([t,cosine])
            # else:
                # print('FOUND',tuple([s, t]))
        targets_cosine = sorted(targets_cosine,key=lambda t: t[1],reverse=similar)
        # print(targets_cosine)
        negative_edges.append([s,targets_cosine[0][0]])
    negative_edges = torch.tensor(negative_edges).t()
    # print(negative_edges)
    # print(edges_t)
    return negative_edges


def random_negative_samples(edge_index, edge_label_index):
    edges_t = edge_index.t().tolist()
    edges_t = [tuple(t) for t in edges_t]

    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    new_edges_t = torch.tensor([[], []])
    found = False
    while not found:
        new_sources = random.sample(sources, edge_label_index.size(1))
        new_targets = random.sample(targets, edge_label_index.size(1))
        new_edges_t = torch.tensor([new_sources, new_targets]).t().tolist()
        new_edges_t = [tuple(t) for t in new_edges_t]
        if set(edges_t).isdisjoint(new_edges_t):
            found = True
    new_edges = torch.tensor([list(e) for e in new_edges_t])
    new_edges = new_edges.t()
    return new_edges



# data = ScholarlyDataset(root='datasets/pubmed/split_transductive/train/')
# data = ScholarlyDataset(root='datasets/mes/split_inductive_full/validation/')
# data = ScholarlyDataset(root='datasets/mes/split_inductive_full/test/')
# import os
#
# data = ScholarlyDataset(root='datasets/pubmed/split_transductive/train/')
# print(data[0])
# print(data[0]['publication'].x[[0,2,3]])


# print(data[0]['publication'].mapping['p_0'])

# file_path = f"model/data/mes/embeddings/publications_net_embeddings.pkl"
# # Apri il file in modalità di lettura binaria ("rb")
# with open(file_path, "rb") as fIn:
#     publications = pickle.load(fIn)
#     print(publications['embeddings'][0].shape)
# # fIn.close()
# for file in os.listdir('datasets/mes/split_inductive_light/validation/'):
#     if 'csv' in file:
#         print(file,pd.read_csv('datasets/mes/split_inductive_light/validation/'+file).shape[0])
#
# dataset = 'mes'
# pubdata_test = pd.read_csv(f'./datasets/{dataset}/split_transductive/test/pubdataedges.csv')
# pubdata_vali = pd.read_csv(f'./datasets/{dataset}/split_transductive/validation/pubdataedges.csv')
# pubdata_train = pd.read_csv(f'./datasets/{dataset}/split_transductive/train/pubdataedges.csv')
#
# pubdata_test_tups = [(row['source'], row['target']) for i, row in pubdata_test.iterrows()]
# pubdata_vali_tups = [(row['source'], row['target']) for i, row in pubdata_vali.iterrows()]
#
# train_forbidden_pubs = list(set(pubdata_test['source'].unique().tolist() + pubdata_vali['source'].unique().tolist()))
# train_forbidden_data = list(set(pubdata_test['target'].unique().tolist() + pubdata_vali['target'].unique().tolist()))
# print(train_forbidden_pubs[0:10])
# print('\n\n')
# for file in os.listdir('datasets/mes/split_inductive_light/validation/'):
#     if 'csv' in file:
#         print(file,pd.read_csv('datasets/mes/split_inductive_light/test/'+file).shape[0])
# print('\n\n')

# for file in os.listdir('datasets/mes/split_transductive/train/'):
#     if 'csv' in file:
#         print(file,pd.read_csv('datasets/mes/split_transductive/train/'+file).shape[0])





