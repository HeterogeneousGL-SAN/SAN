# Code for: Heterogeneous Graph Representation for Dataset Link Prediction and Recommendation on Dynamic and Sparse Scholarly Graphs
This repository contains the code of SAN, proposed at WSDM'25.
MES and PubMed datasets are anonymous and available on [Figshare](https://figshare.com/s/1e11a6f03fbf97d61936)

## Project structure
- In the folder [utils](utils/) there are the file necessary to create the docker image and run the container.
- The folder [preprocessing](preprocessing/) contains the scripts to preprocess data and make these files ready for the augmentation, sampling and aggregation phases. Please, note that the files available on Figshare are already processed.
- The folder [augmentation](augmentation/) contains the scripts to perform entity linking and topic modelling and to analyze the data.
- The folder [split](split/) contains the scripts needed to partition data into training, validation and test (transductive, inductive, semi-inductive) sets.
