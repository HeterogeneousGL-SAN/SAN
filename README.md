# Code for: Heterogeneous Graph Representation for Dataset Link Prediction and Recommendation on Dynamic and Sparse Scholarly Graphs
This repository contains the code of SAN, proposed at WSDM'25.
MES and PubMed datasets are anonymous and available on [Figshare](https://figshare.com/s/1e11a6f03fbf97d61936)

## Project structure
- In the folder `utils` there are the file necessary to create the docker image and run the container.
- The folder [preprocessing` contains the scripts to preprocess data and make these files ready for the augmentation, sampling and aggregation phases. Please, note that the files available on Figshare are already processed.
- The folder `augmentation` contains the scripts to perform entity linking and topic modelling and to analyze the data.
- The folder `split` contains the scripts needed to partition data into training, validation and test (transductive, inductive, semi-inductive) sets.
- The folder `baselines` contains the implementation of the baselines (HAN, HGT, SAGE, GAT, ST-T)
- The folder `model` contains the implementation of SAN comprising the sampling implemented via random walk, and aggregation implemented with multihead attention.
- The folder `additional_analyse` contains other analyses and experiments conducted on different settings wrt those reported in the paper.

## Additional Analyses
Below we report the analyses we performed on SAN and that we did not include in the submitted paper. In particular we analysed: the performances in terms of AUC (link prediction) and ndcg@5, recall@5 (recommendation) of the baselines run with the augmented graph; the analyses of different aggregation approaches.

## Additional experiments on baselines
We evaluated the baselines in two settings: considering the graphs before and after augmentation. While in the paper we reported the results with the original graph, in the tables and plot below we report the results of the baselines run with the augmented graph. The results are reported for 100% and 0% of available metadata. The corresponding AUC plot for MES is in `additional_analyses/mes_aug.pdf` and for PubMed in `additional_analyses/pubmed_aug.pdf`. The last column includes the results of SAN (also reported in the submitted paper).

| MES (%) | Setting | Metric | SAGE  | GAT   | HGT   | HAN   | HGNN  | SAN   |
|---------|---------|--------|-------|-------|-------|-------|-------|-------|
| 100%    | Tran    | R@5    | 0.254 | 0.168 | 0.027 | 0.004 | 0.214 | 0.439 |
|         |         | N@5    | 0.166 | 0.114 | 0.018 | 0.005 | 0.140 | 0.339 |
|         | Semi    | R@5    | 0.266 | 0.229 | 0.044 | 0.000 | 0.237 | 0.428 |
|         |         | N@5    | 0.179 | 0.144 | 0.015 | 0.000 | 0.169 | 0.331 |
|         | Ind     | R@5    | 0.188 | 0.427 | 0.000 | 0.000 | 0.226 | 0.432 |
|         |         | N@5    | 0.132 | 0.298 | 0.000 | 0.000 | 0.153 | 0.345 |
| 0%      | Tran    | R@5    | 0.171 | 0.177 | 0.000 | 0.009 | 0.178 | 0.305 |
|         |         | N@5    | 0.106 | 0.118 | 0.000 | 0.007 | 0.120 | 0.161 |
|         | Semi    | R@5    | 0.152 | 0.140 | 0.000 | 0.000 | 0.156 | 0.301 |
|         |         | N@5    | 0.093 | 0.069 | 0.000 | 0.000 | 0.077 | 0.218 |
|         | Ind     | R@5    | 0.030 | 0.194 | 0.000 | 0.000 | 0.070 | 0.273 |
|         |         | N@5    | 0.012 | 0.106 | 0.000 | 0.000 | 0.037 | 0.170 |


| PubMed (%) | Setting | Metric | SAGE  | GAT   | HGT   | HAN   | HGNN  | SAN   |
|------------|---------|--------|-------|-------|-------|-------|-------|-------|
| 100%       | Tran    | R@5    | 0.252 | 0.174 | 0.002 | 0.016 | 0.114 | 0.230 |
|            |         | N@5    | 0.186 | 0.093 | 0.000 | 0.012 | 0.034 | 0.212 |
|            | Semi    | R@5    | 0.204 | 0.135 | 0.002 | 0.016 | 0.062 | 0.207 |
|            |         | N@5    | 0.148 | 0.083 | 0.001 | 0.008 | 0.036 | 0.156 |
|            | Ind     | R@5    | 0.085 | 0.032 | 0.000 | 0.000 | 0.042 | 0.212 |
|            |         | N@5    | 0.022 | 0.008 | 0.000 | 0.000 | 0.024 | 0.146 |
| 0%         | Tran    | R@5    | 0.135 | 0.124 | 0.000 | 0.023 | 0.046 | 0.238 |
|            |         | N@5    | 0.094 | 0.058 | 0.000 | 0.021 | 0.036 | 0.152 |
|            | Semi    | R@5    | 0.082 | 0.091 | 0.000 | 0.010 | 0.031 | 0.146 |
|            |         | N@5    | 0.065 | 0.049 | 0.000 | 0.006 | 0.014 | 0.110 |
|            | Ind     | R@5    | 0.027 | 0.015 | 0.000 | 0.000 | 0.036 | 0.110 |
|            |         | N@5    | 0.019 | 0.011 | 0.000 | 0.000 | 0.014 | 0.090 |


Both in PubMed and MES datasets, the performances of the baselines on the augmented graphs are similar to those of the non-augmented graphs, specifically they differ of at most 0.06.
## SAN Aggregation Analyses
We performed node-type based aggregation with multihead attention. However, we also experimented biLSTM, GRU. For the final aggregation, we experimented multihead attention, biLSTM, mean pooling. Bwloe we report the results on MES dataset with 100% of metadata available. The results in the two sections below refer to the **SAN** model.
### Node type aggregation
Below we report the application of SAN on MES dataset considering different aggregation approaches. In this section we consider the aggregation of type-based embeddings (the one we performed with multihead attention on the submitted paper). We considered the 100% of metadata available.
For these experiments we concatenated all the embeddings after the aggregation. We report the results for dataset recommendation.

| MES (%) | Setting | Metric | biLSTM | GRU  | mh-attention |
|---------|---------|--------|--------|------|--------------|
| 100%    | Tran    | R@5    | 0.365  | 0.297 | 0.439        |
|         |         | N@5    | 0.245  | 0.214 | 0.339        |
|         | Semi    | R@5    | 0.245  | 0.212 | 0.428        |
|         |         | N@5    | 0.165  | 0.145 | 0.331        |
|         | Ind     | R@5    | 0.235  | 0.315 | 0.432        |
|         |         | N@5    | 0.120  | 0.268 | 0.345        |

The most effective method is multihead attention, while GRU is the least effective.

### All aggregation
Below we report the application of SAN on MES dataset considering different aggregation approaches. In this section we consider the aggregation of the embeddings after node-type based aggregation (the one we performed with concatenation on the submitted paper). We considered the 100% of metadata available.
For these experiments we aggregated the node type based embeddings with multihead attention mechanism.

| MES (%) | Setting | Metric | biLSTM | mh-attention | mean   | concat |
|---------|---------|--------|--------|--------------|--------|--------|
| 100%    | Tran    | R@5    | 0.356  | 0.405        | 0.201  | 0.439  |
|         |         | N@5    | 0.254  | 0.303        | 0.198  | 0.339  |
|         | Semi    | R@5    | 0.321  | 0.400        | 0.259  | 0.428  |
|         |         | N@5    | 0.169  | 0.259        | 0.201  | 0.331  |
|         | Ind     | R@5    | 0.298  | 0.324        | 0.171  | 0.432  |
|         |         | N@5    | 0.123  | 0.267        | 0.110  | 0.345  |




## Before running
### Handle the data
Download the data provided at the URL above and unzip them. You should have two folders, one for MES data and the other one for PubMed data. These data are already processed, hence there is no need to run preprocessing scripts. 
Place the data folders in a folder called `datasets`.

### Create the Docker image 
This code relies on docker hence, if you have not installed it in your machine, follow the instructions provided [here](https://docs.docker.com/get-docker/).
Then, go inside the `utils` folder, where the Dockerfile is, and run:

```
docker build -t san_image:latest .
```

This command will create an image installing all the necessary packages and dependencies mentioned in the Dockerfile and requirements.txt.
Please, note that GPUs are needed.

Then, create a network `db_net` (you can choose the name you prefer). This is needed to properly run entity extraction, if you think you are not going to run augmentation, you can avoid this step.
```
docker network create db_net
```

## Run the Experiments
Belowr, the correct order of phases needed to reproduce the experiments. The instructions of each phase can be found in the corresponding folder and are reported below.

__Step 0: Data preprocessing__: In this step the scholarly knowledge graphs are preprocessed and prepared to the next phases. This step is not needed as data are already processed.

__Step 1: Augmentation__: In this step entities and topics are extracted. The instructions can be found in `augmentation` folder. 

__Step 2: Split__: In this step we partition the data in training, validation, test sets. The instructions can be found in `split` folder.

__Step 3: Run__: Once that the three sets are available, it is possible to run the experiments. Instructions can be found inside the `model` folder.

Please, note that the datasets provided on figshare contain all the data needed to run the experiments. Hence, it is possible to start from the **third** step, which consists in training SAN and evaluate its performances.

## Step 0: Data Preprocessing
This step is not needed, as the data provided in Figshare are already preprocessed.
However, we provided in any case the files we used to preprocess our graphs.

## Step 1: Augmentation
### Entity Linking
Entity linking refers to `entity_linking.py` file. To run entity linking run the following command (the dataset passed as argument can be `pubmed` or `mes`:
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 augmentation/entity_linking.py -dataset=pubmed
```
This code is parallelized in order to save time. To set the number of processors you can set `-processors=2`.

### Topic Modelling
Topic modelling refers to `topic_modelling.py` file. Before running the script, create the folder `augmentation/topic_modelling`. To run topic modelling run the following command (the dataset passed as argument can be `pubmed` or `mes`:
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 augmentation/topic_modelling.py -dataset=pubmed -cluster_size=2
```
The cluster size of the paper is 2 however you can set the size you prefer.

*Please, note that these files will overvwrite the actual content of the mes and pubmed datasets folder as entities and topics files are already present together with the related edges. If you plan to change configurations or code, make sure to save a snapshot of your data before running these scripts.*

## Step 2: Split the data
Split script allows to split the datasets into train, validation, test sets (three sets, for transductive, semi inductive, and inductive setups). the files are already partitioned in the datasets provided on figshare. However for reproducibility purposes:
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name split_container -v /path/sanproject/:/code/ san_image:latest python3 split/split.py -dataset=pubmed
```
The argument `dataset` can take: `mes` and `pubmed` datasets

## Step 3: Train and test SAN
### Experiments
In this folder there are all the files needed to reproduce/reuse the code of SAN. The files ending with `_repro` are files that can be used to reproduce the code to our train, validation and test sets. All the other files instead, are files needed for generalizability purposes to new training, validation, test sets.

- `sampler_repro` and `sampler` contain the implementation of random walks, walks selection and neighbors selection for each node tyoe
- `loader_repro` and `loader` contain the code to load the torch geometric dataset(s)
- `model_repro` and `model` contain the implementation of the SAN model -- i.e., the aggregation phase of the pipeline. Other than multihead attention mentioned in the paper as the best approach, the model allows to use also biLSTM, GRU and mean pooling instead of multihead attention and concatenation for embedding aggregation.
- `utils` contains a set of function useful to the model --i.e., early stopping implementation and networkX useful functions
- `args_list` contains the list of arguments that it is possible to set to run the model such as the number of epochs, the number of heads and minibatch size
- `preprocessing` contains the code to create the node2vec based vectors
- `_bootstrapped` files contain the implementation of intermediate metadata scenario -- we ran SAN 10 times and evaluated it 10 times in order to random samples always different sets of datasets without metadata, in order to allow for the highest variability and ensure experiments robustness
- `_inductive` (both `repro` and `gen`) files allow to run the code in inductive (semi and full) setups. This file is useful on in the reproducibility setup
- `main` files contain allow to run the code and train the model.

**Please, note that if you want to use the graphs already available at: `processed/` folder of the datasets on Figshare, you can only only run the first step of the **Preprocessing** section below and jump directly to the **Experiments** part.

### Preprocessing
The very first step is to configure the folders to host the results and the models learnt.

Run the following script:

```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/setup.py -
```


**Please note that the step below is not necessary if you use the graphs already provided in the original datasets of figshare inside the folder `processed`**

To generate the `node2vec` based embeddings, run:

```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/preprocessing.py -dataset=mes 
```
The dataset argument can take: pubmed or mes.

### Reproducibility
To reproduce the experiments in the transductive setup run the following:

```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/main_rw_repro.py -dataset=mes 
```
you can set all the args available in the `args_list.py` file. If nothing is set, the default configuration will be applied. This file runs the transductive setup. To run the inductive setup replace the `main_rw_repro.py` with `main_rw_inductive_repro.py`. In this case set the flag `-inductive` and the inductive type: use `-inductive_type=light` for the semi-inductive configuration and `-inductive_type=full` for the full inductive configuration.

These files will run the training phase. Running on mes will take about 1,5 hours with the default configuration. Setting different hyperparameters from the command line will slow down/speed up the process.

To test the models, add the `-test` argument. 

To train/test without metadata (0% configuration in the paper), add: `-no_metadata` to the command above.

To train and test in bootstrap mode, hence with different portions of metadata available, use the same code above but with the following file: `main_rw_bootstrapped_repro.py` and set `-bootstrap=25` (or 50, or 75) to set the portion of datasets without metadata. This will train the SAN model 10 times with 10 different portions of datasets without metadata. 

### Generalizability 
You can generalize SAN to new train, validation, test sets partitions. 

The files needed to generalize the code are `main.py` and `main_bootstrapped.py` for the intermediate metadata scenario.
The model works in the same way as above, hence, to run without metadata add the `no_metadata` flag and to test the trained model add `-test`. The model will be trained only once and you will have to test it on three distinct test sets (transductive, semi inductive and inductive). The bootstrap mode work the same as above. Hence the command to run the experiments is:

```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/main.py -dataset=mes 
```

To test in inductive mode use:
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/main.py -dataset=mes -inductive_type=full
```

## Baselines
This folder contains the baselines reported in the paper. The code automatically generates the graphs needed to run the experiments. In this the graphs are those BEFORE the augmentation procedure, hence they differ from those provided in `processed` folder.

Before running the baselines, use the command below to generate the folders needed to run the baselines (i.e., the setup).
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name baselines_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 baselines/baselines_setup.py
```

To run the baselines run:
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name baselines_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 baselines/stt.py -dataset=mes 
```

to run the other baselines, replace `stt.py` with the file you prefer starting with `main`.


