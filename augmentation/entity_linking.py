import pandas as pd
import spacy
import spacy_dbpedia_spotlight
import requests
import spacy
import hashlib
import argparse
import numpy as np
import time
import statistics
import networkx as nx
from multiprocessing import Pool
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", default='mes',choices=['mes','pubmed','pubmed_kcore','mes_full'],
                    type=str)
parser.add_argument("-processors", default=25,type=int)
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('dbpedia_spotlight', config={'dbpedia_rest_endpoint': 'http://10.14.129.2:80/rest', 'confidence': 0.75,
                                          'process': 'annotate'})


# url = 'http://10.14.129.2:80/rest/annotate'
# payload = {
#     'text': "President Obama called Wednesday on Congress to extend a tax break for students included in last year's economic stimulus package, arguing that the policy provides more generous assistance.",
#     'confidence': '0.35'
# }
# headers = {
#     'Accept': 'text/turtle'
# }
#
# response = requests.post(url, data=payload, headers=headers)
#
# print(response.text)

# nlp = spacy.load('en_core_web_sm')
# # Use your endpoint: don't put any trailing slashes, and don't include the /annotate path
# nlp.add_pipe('dbpedia_spotlight', config={'dbpedia_rest_endpoint': 'http://10.14.129.2:80/rest','confidence': 0.1,'process': 'annotate'})
# doc = nlp('Google LLC is an American multinational technology company.')
# print([(ent.text, ent.kb_id_, ent._.dbpedia_raw_result['@similarityScore']) for ent in doc.ents])


# nlp = spacy.blank('en')
# nlp.add_pipe('dbpedia_spotlight')

# doc = nlp('Google LLC is an American multinational technology company.')
# print([(ent.text, ent.kb_id_, ent._.dbpedia_raw_result['@similarityScore']) for ent in doc.ents])

def load_docs(dataset):
    print(dataset)


    # if dataset != 'mes':
    publications = pd.read_csv(f'./datasets/{dataset}/all/final/publications.csv')
    # else:
    #     publications = pd.read_csv(f'./datasets/{dataset}/all/final/publications_filtered.csv')

    datasets = pd.read_csv(f'./datasets/{dataset}/all/final/datasets.csv',low_memory=False)

    publications_d = publications['content'].tolist()
    datasets_d = datasets['content'].tolist()
    publications_ids = publications['id'].tolist()
    datasets_ids = datasets['id'].tolist()
    datasets = [(datasets_ids[i],datasets_d[i]) for i in range(len(datasets_ids))]
    publications = [(publications_ids[i],publications_d[i]) for i in range(len(publications_ids))]
    return publications,datasets

def get_entities(document):
    # print(f'working on {document[0]}')
    doc = nlp(document[1])
    entities = [(ent.text, ent.kb_id_, document[0]) for i,ent in enumerate(doc.ents) if ent.kb_id_ != '']
    # entities = [(ent.text, ent.kb_id_, document[0], 'dbpedia_'+hashlib.md5(ent.text.encode()).hexdigest()) for ent in doc.ents]
    return entities



def save_entities(dataset,processors=25):


    publications, datasets = load_docs(dataset)
    documents = publications + datasets


    # all_entities = []
    # for document in documents:
    #     doc = nlp('Google LLC is an American multinational technology company.')
    #     get_entities(document[1])



    num_processes = processors
    sublists = [documents[i:i + 10000] for i in range(0, len(documents), 10000)]
    allents = []
    print(len(sublists))

    # Create a multiprocessing Pool
    c = 0
    for list in sublists:
        st = time.time()
        with Pool(num_processes) as pool:
            print('working on list: ',c)
            c+=1
            # Map the process_document function to each document in parallel
            ents = pool.map(get_entities, list)
            allents.append(ents)
        end = time.time()

        print('sleeping')
        print(str(end-st))
        time.sleep(5)

    print(len(allents))
    all_entities = []
    length = []
    for doc_list in allents:
        for entity_list in doc_list:
            for entity in entity_list:
                    all_entities.append(entity)

    # Flatten the list of lists into a single list
    # for e in all_entities[0:40]:
    #     print(e)
    print(len(all_entities))
    print('\n\n')

 
    publications = pd.read_csv(f'./datasets/{dataset}/all/final/publications.csv')['id'].unique().tolist()

    datasets = pd.read_csv(f'./datasets/{dataset}/all/final/datasets.csv',low_memory=False)['id'].unique().tolist()


    # all_entities = [entity for sublist in all_entities for entity in sublist]
    df = pd.DataFrame()
    df_tmp = pd.DataFrame()
    df_all = pd.DataFrame()
    df_rels_pubs = pd.DataFrame()
    df_rels_data = pd.DataFrame()
    ids = []
    dbids = []
    pubs = []
    desc = []
    ids_tmp = []
    dbids_tmp = []
    pubs_tmp = []
    desc_tmp = []
    sources_p = []
    targets_p = []
    sources_d = []
    targets_d = []
    c= -1
    d = -1
    for e in all_entities:
        d+=1
        # print(e)
        if e[1] != '':
            cur_ind_db = dbids.index(e[1]) if e[1] in dbids else -1
            if cur_ind_db != -1:
                act = cur_ind_db
            else:
                c += 1
                act = c
                ids.append('dbpedia_' + str(act))
                dbids.append(e[1])
                desc.append(e[0].lower())

            if e[2] in datasets:
                sources_d.append(e[2])
                targets_d.append('dbpedia_'+str(act))

            elif e[2] in publications:
                sources_p.append(e[2])
                targets_p.append('dbpedia_'+str(act))
            else:
                print('not found',e[2])
                break

            ids_tmp.append('dbpedia_' + str(act))
            dbids_tmp.append(e[1])
            desc_tmp.append(e[0].lower())
            pubs_tmp.append(e[2])

    print('built entities list',len(ids))
    df['id'] = ids
    df['dbpedia_id'] = dbids
    df['name'] = desc


    df_tmp['id'] = ids_tmp
    df_tmp['pubs'] = pubs_tmp
    df_tmp['dbpedia_id'] = dbids_tmp
    df_tmp['name'] = desc_tmp

    df_rels_pubs['source'] = sources_p
    df_rels_pubs['target'] = targets_p
    df_rels_data['source'] = sources_d
    df_rels_data['target'] = targets_d

    # conto quelli connessi ad almeno 2 outcomes
    df_all['source'] = sources_p + sources_d
    df_all['target'] = targets_p + targets_d
    df_all_count = df_all.groupby('target').size().reset_index(name='Count')
    print('all target',len(df_all_count))

    df_all_count_1 = df_all_count[df_all_count['Count'] == 1]['target'].unique().tolist()
    print('all target count 1',len(df_all_count_1))

    df_all_count_g20 = df_all_count[(df_all_count['Count'] > 20)]['target'].unique().tolist()
    print('all target count > 20',len(df_all_count_g20))

    df_all_count_g1 = df_all_count[(df_all_count['Count'] > 1)]['target'].unique().tolist()
    print('all target count > 1',len(df_all_count_g1))

    percentile = np.percentile(df_all_count['Count'].tolist(),95)
    df_all_count_per = df_all_count[(df_all_count['Count'] > percentile)]['target'].unique().tolist()
    print(f'PERCENTILE all target count > {percentile}',len(df_all_count_per))

    # df_tmp.to_csv(f'./topic_modelling/entities/{dataset}/entities_tmp.csv',index=False)
    df = df[df['id'].isin(df_all_count_g1) & ~df['id'].isin(df_all_count_per)]
    df.to_csv(f'./datasets/{dataset}/all/final/entities.csv',index=False)
    print(df.shape[0])
    df_rels_pubs = df_rels_pubs[df_rels_pubs['target'].isin(df_all_count_g1)  & ~df_rels_pubs['target'].isin(df_all_count_per)].drop_duplicates()
    df_rels_pubs.to_csv(f'./datasets/{dataset}/all/final/pubentedges.csv',index=False)
    print(df_rels_pubs.shape[0])

    df_rels_data = df_rels_data[df_rels_data['target'].isin(df_all_count_g1) & ~df_rels_data['target'].isin(df_all_count_per)].drop_duplicates()
    df_rels_data.to_csv(f'./datasets/{dataset}/all/final/dataentedges.csv',index=False)
    print(df_rels_data.shape[0])

    # authors and venues
def save_authors_and_venues(dataset):
    df_rels_pubs = pd.read_csv(f'./topic_modelling/entities/{dataset}/pubentedges.csv')
    df_rels_data = pd.read_csv(f'./topic_modelling/entities/{dataset}/dataentedges.csv')
    entities = pd.read_csv(f'./topic_modelling/entities/{dataset}/entities.csv')
    pubauthedges = pd.read_csv(f'./datasets/{dataset}/all/final/pubauthedges.csv')
    dataauthedges = pd.read_csv(f'./datasets/{dataset}/all/final/dataauthedges.csv')
    publications = pd.read_csv(f'./datasets/{dataset}/all/final/publications.csv')['id'].unique().tolist()

    if 'mes' == dataset:
        publications = pd.read_csv(f'./datasets/{dataset}/all/final/publications.csv')['id'].unique().tolist()

    datasets = pd.read_csv(f'./datasets/{dataset}/all/final1/datasets.csv')['id'].unique().tolist()
    df_rels_pubs = df_rels_pubs[df_rels_pubs['source'].isin(publications)]
    df_rels_data = df_rels_data[df_rels_data['source'].isin(datasets)]

    df_rels_pubs.drop_duplicates().to_csv(f'./datasets/{dataset}/all/final/pubentedges.csv', index=False)
    df_rels_data.drop_duplicates().to_csv(f'./datasets/{dataset}/all/final/dataentedges.csv', index=False)
    entities.drop_duplicates().to_csv(f'./datasets/{dataset}/all/final/entities.csv', index=False)

    df_rels_pubs.rename(columns={'target': 'target1', 'source': 'source1'}, inplace=True)
    df_rels_data.rename(columns={'target': 'target1', 'source': 'source1'}, inplace=True)

    if dataset != 'mes':
        pubvenedges = pd.read_csv(f'./datasets/{dataset}/all/final/pubvenuesedges.csv')
        pubvenueentedges = pd.merge(pubvenedges, df_rels_pubs, left_on='source', right_on='source1', how='outer')
        pubvenueentedges = pubvenueentedges[['target', 'target1']]
        pubvenueentedges.rename(columns={'target': 'source', 'target1': 'target'}, inplace=True)

        # pubvenueentedges.drop_duplicates().to_csv(f'./topic_modelling/entities/{dataset}/venuesentedges.csv',index=False)
        pubvenueentedges.drop_duplicates().to_csv(f'./datasets/{dataset}/all/final/venuesentedges.csv',index=False)
        print(pubvenueentedges.shape)

    print('inner1')
    st = time.time()
    pubauthedgesentities = pd.merge(pubauthedges, df_rels_pubs, left_on='source', right_on='source1', how='outer')
    pubauthedgesentities = pubauthedgesentities[['target','target1']]
    pubauthedgesentities.rename(columns={'target': 'source', 'target1': 'target'}, inplace=True)
    # pubauthedgesentities.drop_duplicates().to_csv(f'./topic_modelling/entities/{dataset}/pubauthentedges.csv',index=False)
    pubauthedgesentities.drop_duplicates().to_csv(f'./datasets/{dataset}/all/final/pubauthentedges.csv',index=False)
    print(str(time.time()-st))
    print(pubauthedgesentities.shape)
    print('inner2')
    st = time.time()
    dataauthedgesentities = pd.merge(dataauthedges, df_rels_data, left_on='source', right_on='source1', how='outer')
    dataauthedgesentities = dataauthedgesentities[['target', 'target1']]
    dataauthedgesentities.rename(columns={'target': 'source', 'target1': 'target'}, inplace=True)
    # dataauthedgesentities.drop_duplicates().to_csv(f'./topic_modelling/entities/{dataset}/dataauthentedges.csv',index=False)
    dataauthedgesentities.drop_duplicates().to_csv(f'./datasets/{dataset}/all/final/dataauthentedges.csv',index=False)
    print(str(time.time()-st))
    print(dataauthedgesentities.shape)



def analysis(datasets):
    # analyse publications
    # f = open(f'./topic_modelling/entities/analyses.txt', 'w')
    # line = 'Dataset\tMedian outcomes per 1 entity\tMedian entities per 1 outcome\tMedian pubs per 1 entity\tMedian entities per 1 pub\tMedian datasets per 1 entity\tMedian entities per 1 dataset\tMedian authors(p) per 1 entity\tMedian entities per 1 author(p)\tMedian authors(d) per 1 entity\tMedian entities per 1 author(d)\tMedian venues per 1 entity\tMedian entities per 1 venue'
    # f.write(line)

    for dataset in datasets:
        print(dataset)

        # pubs_source = pd.read_csv(f'./topic_modelling/entities/{dataset}/pubentedges.csv')['source'].tolist()
        # ents_targets = pd.read_csv(f'./topic_modelling/entities/{dataset}/pubentedges.csv')['target'].tolist()
        if dataset == 'mes':
            pubs = pd.read_csv(f'./datasets/{dataset}/all/final/publications.csv')
        else:
            pubs = pd.read_csv(f'./datasets/{dataset}/all/final/publications.csv')

        dats = pd.read_csv(f'./datasets/{dataset}/all/final/datasets.csv')
        df = pd.read_csv(f'./datasets/{dataset}/all/final/pubentedges.csv')
        pubs_source = df.groupby('source').size().reset_index(name='Count')
        pubs_source = pubs_source['Count'].tolist()
        print(f'total pubs: {pubs.shape[0]}')
        print(f'pubs with at least one entity: {len(pubs_source)}')
        ents_targets = df.groupby('target').size().reset_index(name='Count')
        ents_targets.to_csv(f'./datasets/{dataset}/all/final/pubent_target.csv', index=False)

        ents_targets = ents_targets['Count'].tolist()

        # pubs_source = [pubs_source.count(x) for x in set(pubs_source)]
        # ents_targets = [ents_targets.count(x) for x in set(ents_targets)]
        med_ent_x_pub = statistics.median(pubs_source) # median entities per publication
        med_pub_x_ent = statistics.median(ents_targets) # median publications per entity
        print(f'median entities per publication {str(med_ent_x_pub)}')
        print(f'max entities per publication {str(max(pubs_source))}')
        print(f'min entities per publication {str(min(pubs_source))}')
        print(f'median publications per entity {str(med_pub_x_ent)}')
        print(f'max publications per entity {str(max(ents_targets))}')
        print(f'min publications per entity {str(min(ents_targets))}')
        print('\n\n\n')

        df = pd.read_csv(f'./datasets/{dataset}/all/final/dataentedges.csv')
        data_source = df.groupby('source').size().reset_index(name='Count')
        data_source = data_source['Count'].tolist()
        print(f'total data: {dats.shape[0]}')
        print(f'datasets with at least one entity: {len(data_source)}')

        ents_targets = df.groupby('target').size().reset_index(name='Count')
        ents_targets = ents_targets['Count'].tolist()
        med_ent_x_data = statistics.median(data_source) # median entities per publication
        med_data_x_ent = statistics.median(ents_targets) # median publications per entity
        print(f'median entities per dataset {str(med_ent_x_data)}')
        print(f'max entities per dataset {str(max(data_source))}')
        print(f'min entities per dataset {str(min(data_source))}')
        print(f'median datasets per entity {str(med_data_x_ent)}')
        print(f'max datasets per entity {str(max(ents_targets))}')
        print(f'min datasets per entity {str(min(ents_targets))}')
        print('\n\n\n')

        df1 = pd.read_csv(f'./datasets/{dataset}/all/final/dataentedges.csv')
        df0 = pd.read_csv(f'./datasets/{dataset}/all/final/pubentedges.csv')
        df = pd.concat([df1,df0],ignore_index=True)
        data_source = df.groupby('source').size().reset_index(name='Count')
        data_source = data_source['Count'].tolist()
        ents_targets = df.groupby('target').size().reset_index(name='Count')
        ents_targets = ents_targets['Count'].tolist()
        med_ent_x_out = statistics.median(data_source) # median entities per publication
        med_out_x_ent = statistics.median(ents_targets) # median publications per entity
        print(f'median entities per outcome {str(med_ent_x_out)}')
        print(f'max entities per outcome {str(max(data_source))}')
        print(f'min entities per outcome {str(min(data_source))}')
        print(f'median outcomes per entity {str(med_out_x_ent)}')
        print(f'max outcomes per entity {str(max(ents_targets))}')
        print(f'min outcomes per entity {str(min(ents_targets))}\n\n')

        ent = pd.read_csv(f'./datasets/{dataset}/all/final/entities.csv')
        print(f'total entities: {ent.shape[0]}')
        ent = pd.read_csv(f'./datasets/{dataset}/all/final/pubentedges.csv')
        print(f'total pub ent edges: {ent.shape[0]}')
        ent = pd.read_csv(f'./datasets/{dataset}/all/final/dataentedges.csv')
        print(f'total data ent edges: {ent.shape[0]}')


# Use your endpoint: don't put any trailing slashes, and don't include the /annotate path
if __name__ == '__main__':
    args = parser.parse_args()

    dataset = args.dataset
    processors = args.processors
    print('dataset',dataset)
    # save_entities(dataset,processors)
    # save_authors_and_venues(dataset)

    analysis([dataset])

