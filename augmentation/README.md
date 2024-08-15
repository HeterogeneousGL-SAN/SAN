## Entity Linking
Entity linking refers to `entity_linking.py` file. To run entity linking run the following command (the dataset passed as argument can be `pubmed` or `mes`:
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 augmentation/entity_linking.py -dataset=pubmed
```
