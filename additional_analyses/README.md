# Additional experiments
In this section we add additional eperiments we performed but we did not add in the submitted paper. In particular we analysed: the recall and ndcg at cutoffs 1 and 10 for MES and PubMed datasets; the performances in terms of AUC (link prediction) and ndcg@5, recall@5 (recommendation) of the baselines run with the augmented graph; the analyses of different aggregation approaches.

## Dataset recommendation
We evaluated SAN at the following cutoffs: 1, 5 (reported as results in the paper) and 10. Here we report the cutoffs 1 and 10 to demonstrate the effectiveness of SAN.

## Baselines
We evaluated the baselines in two settings: considering the graphs before and after augmentation. While in the paper we reported the results with the original graph, in the tables and plot below we report the results of the baselines run with the augmented graph. The results are reported for 100% and 0% of available metadata.

| PubMed (%) | Setting | Metric | SAGE  | GAT   | HGT   | HAN   | HGNN  |
|------------|---------|--------|-------|-------|-------|-------|-------|
| 100%       | Tran    | R@5    | 0.252 | 0.174 | 0.002 | 0.016 | 0.114 |
|            |         | N@5    | 0.186 | 0.093 | 0.000 | 0.012 | 0.034 |
|            | Semi    | R@5    | 0.204 | 0.135 | 0.002 | 0.016 | 0.062 |
|            |         | N@5    | 0.148 | 0.083 | 0.001 | 0.008 | 0.036 |
|            | Ind     | R@5    | 0.085 | 0.032 | 0.000 | 0.000 | 0.042 |
|            |         | N@5    | 0.022 | 0.008 | 0.000 | 0.000 | 0.024 |
| 0%         | Tran    | R@5    | 0.135 | 0.124 | 0.000 | 0.023 | 0.046 |
|            |         | N@5    | 0.094 | 0.058 | 0.000 | 0.021 | 0.036 |
|            | Semi    | R@5    | 0.082 | 0.091 | 0.000 | 0.010 | 0.031 |
|            |         | N@5    | 0.065 | 0.049 | 0.000 | 0.006 | 0.014 |
|            | Ind     | R@5    | 0.027 | 0.015 | 0.000 | 0.000 | 0.036 |
|            |         | N@5    | 0.019 | 0.011 | 0.000 | 0.000 | 0.014 |



| MES (%) | Setting | Metric | SAGE  | GAT   | HGT   | HAN   | HGNN  |
|------------|---------|--------|-------|-------|-------|-------|-------|
| 100%       | Tran    | R@5    | 0.254 | 0.168 | 0.027 | 0.004 | 0.214 |
|            |         | N@5    | 0.166 | 0.114 | 0.018 | 0.005 | 0.140 |
|            | Semi    | R@5    | 0.266 | 0.229 | 0.044 | 0.000 | 0.237 |
|            |         | N@5    | 0.179 | 0.144 | 0.015 | 0.000 | 0.169 |
|            | Ind     | R@5    | 0.188 | 0.427 | 0.000 | 0.000 | 0.226 |
|            |         | N@5    | 0.132 | 0.298 | 0.000 | 0.000 | 0.153 |
| 0%         | Tran    | R@5    | 0.171 | 0.177 | 0.000 | 0.009 | 0.178 |
|            |         | N@5    | 0.106 | 0.118 | 0.000 | 0.007 | 0.120 |
|            | Semi    | R@5    | 0.152 | 0.140 | 0.000 | 0.000 | 0.156 |
|            |         | N@5    | 0.093 | 0.069 | 0.000 | 0.000 | 0.077 |
|            | Ind     | R@5    | 0.030 | 0.194 | 0.000 | 0.000 | 0.070 |
|            |         | N@5    | 0.012 | 0.106 | 0.000 | 0.000 | 0.037 |

## Aggregation
We performed node-type based aggregation with multihead attention. However, we also experimented biLSTM, GRU. For the final aggregation, we experimented multihead attention, biLSTM, mean pooling. Bwloe we report the results on MES dataset with 100% of metadata available.
