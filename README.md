# Code for: Heterogeneous Graph Representation for Dataset Link Prediction and Recommendation on Dynamic and Sparse Scholarly Graphs
This repository contains the code of SAN, proposed at WSDM'25.
MES and PubMed datasets are anonymous and available on [Figshare](https://figshare.com/s/1e11a6f03fbf97d61936)

## Project structure
- In the folder [utils](utils/) there are the file necessary to create the docker image and run the container.
- The folder [preprocessing](preprocessing/) contains the scripts to preprocess data and make these files ready for the augmentation, sampling and aggregation phases. Please, note that the files available on Figshare are already processed.
- The folder [augmentation](augmentation/) contains the scripts to perform entity linking and topic modelling and to analyze the data.
- The folder [split](split/) contains the scripts needed to partition data into training, validation and test (transductive, inductive, semi-inductive) sets.
- The folder [baselines](baselines/) contains the implementation of the baselines (HAN, HGT, SAGE, GAT, ST-T)
- The folder [model](model/) contains the implementation of SAN comprising the sampling implemented via random walk, and aggregation implemented with multihead attention.
- The folder [additional_analyses](additional_analyses/) contains other analyses and experiments conducted on different settings wrt those reported in the paper.

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

## Run the experiments
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


