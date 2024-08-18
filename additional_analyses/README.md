# Additional experiments
In this section we add additional eperiments we performed but we did not add in the submitted paper. In particular we analysed: the recall and ndcg at cutoffs 1 and 10 for MES and PubMed datasets; the performances in terms of AUC (link prediction) and ndcg@5, recall@5 (recommendation) of the baselines run with the augmented graph; the analyses of different aggregation approaches.

## Dataset recommendation
We evaluated SAN at the following cutoffs: 1, 5 (reported as results in the paper) and 10. Here we report the cutoffs 1 and 10 to demonstrate the effectiveness of SAN.

## Baselines
We evaluated the baselines in two settings: considering the graphs before and after augmentation. While in the paper we reported the results with the original graph, in the tables and plot below we report the results of the baselines run with the augmented graph.

## Aggregation
We performed node-type based aggregation with multihead attention. However, we also experimented biLSTM, GRU. For the final aggregation, we experimented multihead attention, biLSTM, mean pooling. Bwloe we report the results on MES dataset with 100% of metadata available.
