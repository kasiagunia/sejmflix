# This file collects all functions used for preprocessing from files: read_data.ipynb and data_preprocessing.ipynb

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import networkx as nx


def xml_to_dataframe(xml_file):
    
    """Takes a directory of a xml file and returns a dataframe."""
    
    data = []
    columns = ["kadencja", "posiedzenie", "numer", "data", "godzina", "tryb", "tytul", "temat", 
               "opis", "url", "za", "przeciw", "wstrzym", "niegl", "Nrleg", "NazwiskoImie", "Glos", "Klub"]
    
    with open(xml_file, "r", encoding="utf-8") as f:
        context = ET.iterparse(f, events=("start", "end"))
        _, root = next(context)  # Get root element

        for event, element in context:
            if event == "end" and element.tag == "Glosowanie":
                row_data = {}
                row_data["kadencja"] = element.get("kadencja")
                row_data["posiedzenie"] = element.get("posiedzenie")
                row_data["numer"] = element.get("numer")
                row_data["data"] = element.get("data")
                row_data["godzina"] = element.get("godzina")
                row_data["tryb"] = element.get("tryb")
                row_data["tytul"] = element.find("Tytul").text if element.find("Tytul") is not None else None
                row_data["temat"] = element.find("Temat").text if element.find("Temat") is not None else None
                row_data["opis"] = element.find("Opis").text if element.find("Opis") is not None else None
                row_data["url"] = element.find("url").text if element.find("url") is not None else None

                wyniki = element.find("Wyniki")
                if wyniki:
                    row_data["za"] = wyniki.get("za")
                    row_data["przeciw"] = wyniki.get("przeciw")
                    row_data["wstrzym"] = wyniki.get("wstrzym")
                    row_data["niegl"] = wyniki.get("niegl")
                    for glos_posla in wyniki.findall("GlosPosla"):
                        glosowanie_data = row_data.copy()
                        glosowanie_data["Nrleg"] = glos_posla.find("Nrleg").text if glos_posla.find("Nrleg") is not None else None
                        glosowanie_data["NazwiskoImie"] = glos_posla.find("NazwiskoImie").text if glos_posla.find("NazwiskoImie") is not None else None
                        glosowanie_data["Glos"] = glos_posla.find("Glos").text if glos_posla.find("Glos") is not None else None
                        glosowanie_data["Klub"] = glos_posla.find("Klub").text if glos_posla.find("Klub") is not None else None
                        data.append(glosowanie_data)

                root.clear()  # Clear the root element to free up memory
        
    df = pd.DataFrame(data, columns=columns)
    return df

def save_xml_as_csv(xml_file, csv_file):
    df = xml_to_dataframe(xml_file)
    
    df['data_godzina'] = pd.to_datetime(df['data'] + ' ' + df['godzina'])

    df = df.astype({
                'Nrleg': 'int32',
                'kadencja': 'int32',
                'posiedzenie': 'int32',
                'numer': 'int32',
                })

    df['vote_id'] = df['kadencja'] * 10**6 + df['posiedzenie'] * 10**3 + df['numer']
    df_ = df[['vote_id', 'data_godzina', 'Nrleg', 'NazwiskoImie', 'Glos', 'Klub']] 
    df_.to_csv(csv_file, index=False)
    return df_

def create_deputy_df(df):
    df_deputies = df[['Nrleg', 'NazwiskoImie', 'Klub']]
    df_deputies = df_deputies.groupby(['Nrleg', 'NazwiskoImie'])['Klub'].agg(['unique']).reset_index().rename(columns={"unique": "Klub"})
    return df_deputies

def make_queues(df_deputies, vote_ids, deputy_ids_per_vote):
    removed_ids = []
    new_ids = []

    for i in range(1, len(vote_ids)):
        ids_0 = deputy_ids_per_vote[i-1]
        ids_1 = deputy_ids_per_vote[i]

        removed = set(ids_0).difference(ids_1)
        new = set(ids_1).difference(ids_0)

        if len(removed) > 0:
            for idd in removed:
                r = (idd, list(df_deputies[df_deputies['Nrleg'] == idd]['Klub'].values[0]), vote_ids[i])
                removed_ids.append(r)
        if len(new) > 0:
            for idd in new:
                n = (idd, list(df_deputies[df_deputies['Nrleg'] == idd]['Klub'].values[0]), vote_ids[i])
                new_ids.append(n)
    return removed_ids, new_ids

def find_pairs(removed_ids, new_ids):
    ids_pairs = []
    node_id_dict = {i:i for i in range(1, 461)}

    while len(removed_ids) > 0:
        r_id, r_c, r_v_id = removed_ids.pop(0)
        search = 1
        i = 0
        while search and i < len(new_ids):
            n_id, n_c, n_v_id = new_ids[i]
            if len(set(r_c).intersection(set(n_c))) > 0 and r_v_id <= n_v_id:
                ids_pairs.append((r_id, n_id))
                node_id_dict[n_id] = r_id
                new_ids.pop(i)
                search = 0
            i += 1
    return node_id_dict

def assign_node_ids(df):
    df_deputies = create_deputy_df(df)
    
    vote_ids = sorted(df['vote_id'].unique())
    deputy_ids_per_vote = [df[df['vote_id'] == vote_id]['Nrleg'].unique() for vote_id in vote_ids]
    
    removed_ids, new_ids = make_queues(df_deputies, vote_ids, deputy_ids_per_vote)
    node_id_dict = find_pairs(removed_ids, new_ids)
    
    node_id_dict_func = lambda x: node_id_dict[x]
    node_id_dict_func = np.vectorize(node_id_dict_func)
    
    df_deputies['node_id'] = node_id_dict_func(df_deputies['Nrleg'])
    df_deputies['node_id'] = df_deputies['node_id'] - 1
        
    return df_deputies

def assign_attributes(df):
    df_node_atr = df.groupby('node_id').agg({'Nrleg': 'unique',
                                             'NazwiskoImie': pd.Series.unique, 
                                             'Klub': pd.Series.unique})
    df_node_atr['Klub'] = df_node_atr['Klub'].apply(lambda x: x[-1])
    return df_node_atr.reset_index()

def egde_weights(df):
    """Returns matrix with edge values."""
    common_votes = np.zeros((460, 460))
    vote_ids = df['vote_id'].unique()
    
    for vote_id in tqdm(vote_ids):
        dep_yes = df[(df['vote_id'] == vote_id) & (df['Glos'] == 'Za')]['node_id'].values
        dep_no = df[(df['vote_id'] == vote_id) & (df['Glos'] == 'Przeciw')]['node_id'].values
        dep_abstain = df[(df['vote_id'] == vote_id) & (df['Glos'] == 'Wstrzymał się')]['node_id'].values
        dep_no_vote = df[(df['vote_id'] == vote_id) & (df['Glos'] == 'Nie oddał głosu')]['node_id'].values
#         dep_absent = df[(df['vote_id'] == vote_id) & (df['Glos'] == 'Nieobecny')]['node_id'].values

        # votes in favor
        for i in range(len(dep_yes)):
            for j in range(i+1, len(dep_yes)):
                common_votes[dep_yes[i], dep_yes[j]] += 1
                common_votes[dep_yes[j], dep_yes[i]] += 1

        # votes against
        for i in range(len(dep_no)):
            for j in range(i+1, len(dep_no)):
                common_votes[dep_no[i], dep_no[j]] += 1
                common_votes[dep_no[j], dep_no[i]] += 1

        # abstain from vote
        for i in range(len(dep_abstain)):
            for j in range(i+1, len(dep_abstain)):
                common_votes[dep_abstain[i], dep_abstain[j]] += 1
                common_votes[dep_abstain[j], dep_abstain[i]] += 1
                
        # didn't vote
        for i in range(len(dep_no_vote)):
            for j in range(i+1, len(dep_no_vote)):
                common_votes[dep_no_vote[i], dep_no_vote[j]] += 1
                common_votes[dep_no_vote[j], dep_no_vote[i]] += 1

    return common_votes / len(vote_ids)

def create_graph(edge_matrix, df_node_atr, file_name=None, monthly=True):
    G = nx.from_numpy_array(edge_matrix)
    
#     for n1, n2, e_weight in G.edges.data('weight'):
#         G.edges[n1, n2]['distance'] = 1 / e_weight
        
    for i in range(460):
        row = df_node_atr[df_node_atr['node_id'] == i]
        if row.size > 0:
            node_id, nr_leg, name, party = row.values[0]
        else:
            print(node_id, file_name) # bug: it prints (node_id-1) node
            party = ''
        G.nodes[i]['party'] = party
    
    if file_name:
        if monthly:
            pickle.dump(G, open(f'graphs/monthly/{file_name}.pickle', 'wb'))
            nx.write_graphml(G, f"graphs_graphml/monthly/{file_name}.graphml")
        else:
            pickle.dump(G, open(f'graphs/{file_name}.pickle', 'wb'))
            nx.write_graphml(G, f"graphs_graphml/{file_name}.graphml")
    return G