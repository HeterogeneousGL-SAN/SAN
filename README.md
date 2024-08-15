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

## Before running
### Handle the data
Download the data provided at the URL above and unzip them. You should have two folders, one for MES data and the other one for PubMed data. These data are already processed, hence there is no need to run preprocessing scripts. 
Place the data folders in a folder called `datasets`.

### Create the Docker image 
This code relies on docker hence, if you have not installed it in your machine, follow the instructions provided (here)[https://docs.docker.com/get-docker/].
Then, go inside the `utils` folder, where the Dockerfile is, and run:

```
docker build -t san_image:latest .
```

This command will create an image installing all the necessary packages and dependencies mentioned in the Dockerfile and requirements.txt.
Please, note that GPUs are needed.






