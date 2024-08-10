import os
import torch.nn as nn
import args_list
from sampler import RandomWalkWithRestart
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric import seed_everything
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
seed_everything(42)

class CosineSimilarityAttention(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super(CosineSimilarityAttention, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, query, keys):
        # Ensure query has the same batch size as keys
        if query.size(0) == 1:
            query = query.expand(keys.size(0), -1, -1)

        # Normalize the query and keys
        query_norm = query / query.norm(dim=-1, keepdim=True)  # (batch_size, seq_length, input_dim)
        keys_norm = keys / keys.norm(dim=-1, keepdim=True)  # (batch_size, seq_length, input_dim)
        query_norm = query_norm.transpose(1, 2)
        attention_scores = torch.matmul(keys_norm, query_norm)  # (batch_size, seq_length, seq_length)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, seq_length, seq_length)
        output = torch.sum(torch.bmm(attention_weights, keys), dim=0)  # (batch_size, input_dim)

        x = self.relu(self.fc(output))
        return x, attention_weights


class ScHetGNN(nn.Module):
    def __init__(self,args):
        super(ScHetGNN, self).__init__()
        self.args = args

        # if self.args.train:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # else:
        # self.device = 'cpu'
        # print(self.device)
        self.lstm_core_aggregator = nn.LSTM(args.core_dim, int(args.embedding_dim/2), args.lstm_layers, bidirectional=True)
        self.lstm_core_net_aggregator = nn.LSTM(args.top_dim, int(args.embedding_dim/2), args.lstm_layers, bidirectional=True)
        self.lstm_key_aggregator = nn.LSTM(args.key_dim,  int(args.embedding_dim/2), args.lstm_layers, bidirectional=True)
        self.lstm_key_net_aggregator = nn.LSTM(args.top_dim,  int(args.embedding_dim/2), args.lstm_layers, bidirectional=True)
        self.lstm_top_aggregator = nn.LSTM(args.top_dim,  int(args.embedding_dim/2), args.lstm_layers, bidirectional=True)
        self.lstm_aggregator = nn.LSTM(args.embedding_dim, int(args.embedding_dim/2), args.lstm_layers, bidirectional=True)

        self.gru_core_aggregator = nn.GRU(args.core_dim, args.embedding_dim, args.gru_layers)
        self.gru_key_aggregator = nn.GRU(args.key_dim, args.embedding_dim, args.gru_layers)
        self.gru_top_aggregator = nn.GRU(args.top_dim, args.embedding_dim, args.gru_layers)
        self.gru_aggregator = nn.GRU(args.embedding_dim, args.embedding_dim, args.gru_layers)


        self.multihead_core_attn = nn.MultiheadAttention(args.core_dim, self.args.heads)
        self.multihead_core_net_attn = nn.MultiheadAttention(args.top_dim, self.args.heads)
        self.multihead_key_attn = nn.MultiheadAttention(args.key_dim, self.args.heads)
        self.multihead_key_net_attn = nn.MultiheadAttention(args.top_dim, self.args.heads)
        self.multihead_top_attn = nn.MultiheadAttention(args.top_dim,self.args.heads)
        self.multihead_attn = nn.MultiheadAttention(args.embedding_dim, self.args.heads)

        self.attention_core = CosineSimilarityAttention(args.core_dim, args.embedding_dim)
        self.attention_key = CosineSimilarityAttention(args.key_dim, args.embedding_dim)
        self.attention_top = CosineSimilarityAttention(args.top_dim, args.embedding_dim)

        self.core_linear_projection = nn.Linear(args.core_dim, args.embedding_dim)
        self.core_net_linear_projection = nn.Linear(args.top_dim, args.embedding_dim)
        self.key_linear_projection = nn.Linear(args.key_dim, args.embedding_dim)
        self.key_net_linear_projection = nn.Linear(args.top_dim, args.embedding_dim)
        self.top_linear_projection = nn.Linear(args.top_dim, args.embedding_dim)

        self.act = torch.nn.LeakyReLU()



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter) or isinstance(m, nn.LSTM)  or isinstance(m, nn.GRU)  or isinstance(m, nn.MultiheadAttention):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)



    def aggregate_same_type_neigh(self,seed_vectors,neigh_embeddings,agg_type,type_emb='core'):
        embeddings = [lista for i, lista in enumerate(neigh_embeddings)]  # appendo il target vector, utile nell'aggregazione finale
        queries = []
        for ind, e in enumerate(embeddings):
            seed = seed_vectors[ind]
            queries.append(seed)
        queries = [[emb[-1]] for emb in queries]
        queries_tensors = [torch.stack(t) for t in queries]
        embeddings_tensors = [[emb[-1] for emb in lista] for lista in embeddings]
        embeddings_tensors = [torch.stack(t) for t in embeddings_tensors]

        concatenated_queries_matt = torch.stack(queries_tensors, dim=0)
        concatenated_embeddings_matt = torch.stack(embeddings_tensors, dim=0)
        concatenated_embeddings_matt = torch.transpose(concatenated_embeddings_matt, 0, 1).to(self.device)
        concatenated_queries_matt = torch.transpose(concatenated_queries_matt, 0, 1).to(self.device)

        if agg_type == 'mh-attention' and (type_emb == 'core' or type_emb == 'core_net'):
            if type_emb == 'core':
                output, _ = self.multihead_core_attn(concatenated_queries_matt.to(self.device),concatenated_embeddings_matt.to(self.device),concatenated_embeddings_matt.to(self.device))
                output = output[0, :, :]
                output = self.act(self.core_linear_projection(output.to(self.device)))

            if type_emb == 'core_net':
                output, _ = self.multihead_core_net_attn(concatenated_queries_matt.to(self.device),concatenated_embeddings_matt.to(self.device),concatenated_embeddings_matt.to(self.device))
                output = output[0, :, :]
                output = self.act(self.core_net_linear_projection(output.to(self.device)))


            if self.args.verbose:
                print(f'output {output.shape}')

            return output

        elif  agg_type == 'mh-attention':
            if type_emb == 'key':

                output, _ = self.multihead_key_attn(concatenated_embeddings_matt,concatenated_embeddings_matt,concatenated_embeddings_matt)
                output = torch.mean(output, dim=0)
                output = self.act(self.key_linear_projection(output))


            elif type_emb == 'key_net':

                output, _ = self.multihead_key_net_attn(concatenated_embeddings_matt,concatenated_embeddings_matt,concatenated_embeddings_matt)
                output = torch.mean(output, dim=0)
                output = self.act(self.key_net_linear_projection(output))

            else:

                output, _ = self.multihead_top_attn(concatenated_embeddings_matt,concatenated_embeddings_matt,concatenated_embeddings_matt)
                output = torch.mean(output, dim=0)
                output = self.act(self.top_linear_projection(output))

            if self.args.verbose:
                print(f'output {output.shape}')

            return output

        elif agg_type == 'mean_pooling':

            """Se eseguito solo con keep_cores non richiede il training!"""

            embeddings_tensors = [[emb[-1] for emb in lista] for lista in embeddings]

            embeddings_tensors = [torch.stack(t) for t in embeddings_tensors]
            embeddings_tensors = torch.stack(embeddings_tensors, dim=0)

            weights_tensors = [[torch.tensor(emb[2]) for emb in lista] for lista in embeddings]
            weights_tensors = [torch.stack(t) for t in weights_tensors]
            weights_tensors = torch.stack(weights_tensors, dim=0)

            weighted_sum = torch.sum(embeddings_tensors * weights_tensors.unsqueeze(-1), dim=1)
            weights_sum = torch.sum(weights_tensors, dim=1)
            weighted_avg = weighted_sum / weights_sum.unsqueeze(-1)
            return weighted_avg

        elif agg_type == 'lstm':
            if type_emb == 'core':
                output, _ = self.lstm_core_aggregator(concatenated_embeddings_matt.to(self.device))
            elif type_emb == 'core_net':
                output, _ = self.lstm_core_net_aggregator(concatenated_embeddings_matt.to(self.device))
            elif type_emb == 'key':
                output, _ = self.lstm_key_aggregator(concatenated_embeddings_matt.to(self.device))
            elif type_emb == 'key_net':
                output, _ = self.lstm_key_net_aggregator(concatenated_embeddings_matt.to(self.device))
            elif type_emb == 'top':
                output, _ = self.lstm_top_aggregator(concatenated_embeddings_matt.to(self.device))

            if self.args.verbose:
                print(f'output lstm {output.shape}')

            return torch.mean(output,0)

        elif agg_type == 'gru':
            if type_emb == 'core':
                output, _ = self.gru_core_aggregator(concatenated_embeddings_matt.to(self.device))
            elif type_emb == 'key':
                output, _ = self.gru_key_aggregator(concatenated_embeddings_matt.to(self.device))
            elif type_emb == 'top':
                output, _ = self.gru_top_aggregator(concatenated_embeddings_matt.to(self.device))

            if self.args.verbose:
                print(f'output {output.shape}')
            return torch.mean(output,0)



    def combine_all(self,core_emb,key_emb,top_emb,cores_net_emb=None,key_net_emb=None,aggr_type='concat'):
        if aggr_type == 'mh-attention':

            embeddings = [emb for emb in [core_emb, key_emb, top_emb] if emb is not None]
            if cores_net_emb is not None and len(cores_net_emb)>0:
                embeddings = [emb for emb in [cores_net_emb,core_emb,key_emb,top_emb] if emb is not None]
            if key_net_emb is not None and len(key_net_emb)>0:
                embeddings = [emb for emb in [cores_net_emb,core_emb,key_net_emb,key_emb,top_emb] if emb is not None]

            output_stacked = torch.stack(embeddings,dim=0).to(self.device)
            output, attention_weights = self.multihead_attn(output_stacked, output_stacked, output_stacked)
            output = torch.mean(output,0)
            return output

        elif aggr_type == 'mean':
            embeddings = [emb for emb in [core_emb, key_emb, top_emb] if emb is not None]
            concat_emb = torch.stack(embeddings, dim=0)
            output = torch.mean(concat_emb, 0)
            return output

        elif aggr_type == 'lstm':
            embeddings = [emb for emb in [core_emb, key_emb, top_emb] if emb is not None]
            concat_emb = torch.stack(embeddings, dim=0)
            output, _ = self.lstm_aggregator(concat_emb.to(self.device))
            return torch.mean(output,0)

        elif aggr_type == 'gru':
            embeddings = [emb for emb in [core_emb,key_emb,top_emb] if emb is not None]
            concat_emb = torch.stack(embeddings, dim=0)
            output, hnn = self.gru_aggregator(concat_emb.to(self.device))
            return torch.mean(output,0)

        elif aggr_type == 'concat':
            embeddings = [emb for emb in [core_emb, key_emb, top_emb] if emb is not None]
            if cores_net_emb is not None and len(cores_net_emb)>0:
                embeddings = [emb for emb in [cores_net_emb,core_emb,key_emb,top_emb] if emb is not None]
            if key_net_emb is not None and len(key_net_emb)>0:
                embeddings = [emb for emb in [cores_net_emb,core_emb,key_net_emb,key_emb,top_emb] if emb is not None]

            emb = torch.cat(embeddings,dim=1)
            return emb


    def forward(self,vectors,vectors_net,net_core_embeddings,core_embeddings,net_key_embeddings,key_embeddings,top_embeddings,core_agg='lstm',top_agg='lstm',key_agg='lstm',all_agg='concat'):
        key_embeddings = self.aggregate_same_type_neigh(vectors,key_embeddings,key_agg,type_emb='key')
        core_embeddings = self.aggregate_same_type_neigh(vectors,core_embeddings,core_agg,type_emb='core')
        top_embeddings = self.aggregate_same_type_neigh(vectors,top_embeddings,top_agg,type_emb='top')

        if self.args.verbose:
            print(f'final cores dimension {core_embeddings.shape}')
            print(f'final key dimension {key_embeddings.shape}')
            print(f'final tops dimension {top_embeddings.shape}')


        if self.args.enriched == 'enriched':
            core_net_embeddings = self.aggregate_same_type_neigh(vectors_net, net_core_embeddings, top_agg,
                                                                 type_emb='core_net')
            final_embeddings = self.combine_all(core_embeddings,key_embeddings,top_embeddings,core_net_embeddings,aggr_type=all_agg)
        elif self.args.enriched == 'enriched_all':
            core_net_embeddings = self.aggregate_same_type_neigh(vectors_net, net_core_embeddings, top_agg,
                                                                 type_emb='core_net')
            key_net_embeddings = self.aggregate_same_type_neigh(vectors, net_key_embeddings, top_agg, type_emb='key_net')
            final_embeddings = self.combine_all(core_embeddings,key_embeddings,top_embeddings,core_net_embeddings,key_net_embeddings,aggr_type=all_agg)
        elif self.args.enriched == 'standard':
            final_embeddings = self.combine_all(core_embeddings,key_embeddings,top_embeddings,aggr_type=all_agg)

        if self.args.verbose:
            print(f'final emb dimension {final_embeddings.shape}')
        return final_embeddings

    def cross_entropy_loss(self,embeddings,pos_source,pos_target,neg_source,neg_target):
        pos_embeddings_source,neg_embeddings_source = embeddings[pos_source],embeddings[neg_source]
        pos_embeddings_target,neg_embeddings_target = embeddings[pos_target],embeddings[neg_target]
        # print('LOSS')
        # print(pos_embeddings_source)
        pos_embeddings_source = pos_embeddings_source.view(pos_embeddings_source.size(0), 1, pos_embeddings_source.size(1))  # [batch_size, 1, embed_d]
        neg_embeddings_source = neg_embeddings_source.view(neg_embeddings_source.size(0), 1, neg_embeddings_source.size(1))  # [batch_size, 1, embed_d]
        pos_embeddings_target = pos_embeddings_target.view(pos_embeddings_target.size(0), pos_embeddings_target.size(1), 1)  # [batch_size, embed_d, 1]
        neg_embeddings_target = neg_embeddings_target.view(neg_embeddings_target.size(0), neg_embeddings_target.size(1), 1)  # [batch_size, embed_d, 1]

        # Calcola la similarità (prodotto scalare) tra embedding positivi e negativi
        out_p = torch.bmm(pos_embeddings_source, pos_embeddings_target)  # [batch_size, 1, 1]
        out_n = -torch.bmm(neg_embeddings_source, neg_embeddings_target)  # [batch_size, 1, 1]
        sum_p = F.logsigmoid(out_p)
        sum_n = F.logsigmoid(out_n)


        loss_sum = - (sum_p + sum_n)
        return loss_sum.mean()



