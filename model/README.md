# Experiments
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


## Preprocessing
The very first step is to configure the folders to host the results and the models learnt.

Run the following script:

```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/setup.py -
```


To generate the `node2vec` based embeddings, run:

```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/preprocessing.py -dataset=mes -
```
The dataset argument can take: pubmed or mes.



