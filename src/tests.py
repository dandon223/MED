from scipy.io import arff
import os
from typing import List, Tuple, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from nn_chain_linkage import NNChainLinkage, get_clusters
from pyclustering.cluster.cure import cure
from sklearn import metrics

def get_dataset(dataset: str) -> Tuple[pd.DataFrame, int]:
    data = arff.loadarff(dataset)
    df = pd.DataFrame(data[0])
    df.iloc[:, -1] = df.iloc[:, -1].str.decode("utf-8")
    number_of_clusters = df.iloc[:, -1].nunique()
    df = df.rename(columns={df.columns[-1]: "ground_truth"})

    for column in df.columns.values:
        if df[column].dtype == "object":
            df[column] = df[column].astype("category")
            df[column] = df[column].cat.codes

    df.fillna(df.mean(), inplace=True)
    return df, number_of_clusters

def get_datasets(files: List[Path]) -> Tuple[Dict, Dict]:
    datasets_dict = {}
    numbers_of_clusters_dict = {}
    for file in files:
        try:
            df, number_of_clusters = get_dataset(file)
        except (AttributeError, ValueError) as e:
            print(f"Did not load the file: {file}, as it did not have cluster labels as the last column")
            continue
        except NotImplementedError as e:
            print(f"Did not load the file: {file}, as it is unsupported by ARFF library")
            continue
        file_name = os.path.basename(file)
        datasets_dict[file_name] = df
        numbers_of_clusters_dict[file_name] = number_of_clusters
    return datasets_dict, numbers_of_clusters_dict

def test() -> None:

    print("test_rand_score_artificial")
    test_rand_score("../datasets/artificial", "../results/rand_score_artificial.csv")
    print("test_rand_score_real-world")
    test_rand_score("../datasets/real-world", "../results/real_world.csv")

    return 0

def test_rand_score(folder: str, csv_name: str):
    datasets, number_of_clusters = get_datasets(Path(folder).glob("*"))

    print("test_agglomerative_clustering")
    df = test_agglomerative_clustering(datasets, number_of_clusters)
    print("test_cure_clustering")
    df_cure = test_cure_clustering(datasets, number_of_clusters)
    df["dataset_name2"] = df_cure["dataset_name"]
    df["cure_clustering_rand_score"] = df_cure["cure_clustering_rand_score"]

    df.to_csv(csv_name, index=False)

def test_cure_clustering(datasets: Dict[str, pd.DataFrame], number_of_clusters: Dict[str, int]) -> pd.DataFrame:
    results = []
    for dataset in datasets:
        result = {}
        data = datasets[dataset]
        Y_data = data["ground_truth"]
        X_data = data.drop(['ground_truth'], axis=1)
        cure_instance = cure(X_data.to_numpy() , number_of_clusters[dataset])
        cure_instance.process()
        clusters = cure_instance.get_clusters()
        data["cure_clustering"] = np.nan
        for index, cluster in enumerate(clusters):
            for point in cluster:
                data.at[point, "cure_clustering"] = index
        rand_score = metrics.rand_score( labels_true=Y_data, labels_pred=data["cure_clustering"])
        result["dataset_name"] = dataset
        result["cure_clustering_rand_score"] = rand_score
        results.append(result)
    df = pd.DataFrame(results)
    return df

def test_agglomerative_clustering(datasets: Dict[str, pd.DataFrame], number_of_clusters: Dict[str, int]) -> pd.DataFrame:
    results = []
    for dataset in datasets:
        hierarchical_cluster = AgglomerativeClustering(n_clusters=number_of_clusters[dataset], affinity='euclidean', linkage='single')
        result = {}
        data = datasets[dataset]
        Y_data = data["ground_truth"]
        X_data = data.drop(['ground_truth'], axis=1)
        labels = hierarchical_cluster.fit_predict(X_data)
        rand_score = metrics.rand_score( labels_true=Y_data, labels_pred=labels)
        result["dataset_name"] = dataset
        result["agglomerative_clustering_rand_score"] = rand_score
        results.append(result)
    df = pd.DataFrame(results)
    return df

def dev_test(dataset: str = "../datasets/artificial/target.arff") -> None:
    data, number_of_clusters = get_dataset(dataset)
    Y_data = data["ground_truth"]
    X_data = data.drop(['ground_truth'], axis=1)

    hierarchical_cluster = AgglomerativeClustering(n_clusters=number_of_clusters, affinity='euclidean', linkage='ward')
    labels = hierarchical_cluster.fit_predict(X_data)
    rand_score = metrics.rand_score( labels_true=Y_data, labels_pred=labels)
    print('rand_score', rand_score)

    cure_instance = cure(X_data.to_numpy() , number_of_clusters)
    cure_instance.process()
    clusters = cure_instance.get_clusters()

    data["CUREClustering"] = Y_data
    for index, cluster in enumerate(clusters):
        for point in cluster:
            data.at[point, "CUREClustering"] = index
    rand_score = metrics.rand_score( labels_true=Y_data, labels_pred=data["CUREClustering"])
    print('rand_score', rand_score)

    data["AgglomerativeClustering"] = labels


    nn_chain_linkage_alg = NNChainLinkage()
    linkage = nn_chain_linkage_alg.fit_predict(X_data.to_numpy())
    clusters = get_clusters(linkage, len(X_data), number_of_clusters)
    for index, cluster in enumerate(clusters):
        for point in cluster:
            data.at[point, "CUREClustering"] = index

    rand_score = metrics.rand_score( labels_true=Y_data, labels_pred=data["CUREClustering"])
    print('rand_score', rand_score)
    print("px.scatter")
    fig = px.scatter(data,
                         x='x',
                         y='y',
                         color='ground_truth',
                         title=f"GroundTruth {dataset}")
    fig.show()

    fig = px.scatter(data,
                         x='x',
                         y='y',
                         color='AgglomerativeClustering',
                         title=f"AgglomerativeClustering {dataset}")
    fig.show()

    fig = px.scatter(data,
                         x='x',
                         y='y',
                         color='CUREClustering',
                         title=f"my {dataset}")
    fig.show()
