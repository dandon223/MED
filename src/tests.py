from scipy.io import arff
import os
from typing import List, Tuple, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import time
from sklearn.cluster import AgglomerativeClustering
from nn_chain_linkage import NNChainLinkage, get_clusters
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
    df.drop_duplicates(inplace=True, ignore_index=True)
    #d = df.duplicated(subset=['left-weight','left-distance', 'right-weight', 'right-distance'])
    #df["yes"] = d
    #df.to_csv("t.csv")
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

    #print("single_test_rand_score_artificial")
    #test_rand_score_time("../datasets/artificial-subset", "../results/single_rand_score_artificial-subset.csv", "euclidean", "single", 5)
    #print("single_test_rand_score_real_world_subset")
    #test_rand_score_time("../datasets/real-world-subset", "../results/single_rand_score_real_world_subset.csv", "euclidean", "single", 5)

    #print("complete_test_rand_score_artificial")
    #test_rand_score_time("../datasets/artificial-subset", "../results/complete_rand_score_artificial-subset.csv", "euclidean", "complete", 5)
    #print("complete_test_rand_score_real_world_subset")
    #test_rand_score_time("../datasets/real-world-subset", "../results/complete_rand_score_real_world_subset.csv", "euclidean", "complete", 5)

    #print("average_test_rand_score_artificial")
    #test_rand_score_time("../datasets/artificial-subset", "../results/average_score_artificial-subset.csv", "euclidean", "average", 5)
    #print("average_test_rand_score_real_world_subset")
    #test_rand_score_time("../datasets/real-world-subset", "../results/average_score_real_world_subset.csv", "euclidean", "average", 5)

    #print("single_manhattan_test_rand_score_artificial")
    #test_rand_score_time("../datasets/artificial-subset", "../results/single_manhattan_rand_score_artificial-subset.csv", "manhattan", "single", 5)
    #print("single_manhattan_test_rand_score_real_world_subset")
    #test_rand_score_time("../datasets/real-world-subset", "../results/single_manhattan_rand_score_real_world_subset.csv", "manhattan", "single", 5)

    #print("number_of_features_test")
    #columns_number = [2, 4, 8, 16, 32, 64, 128, 256]
    #test_number_of_features(columns_number, number_of_times=5, column_name="nn_linkage_clustering", csv_name="../results/arrhythmia_features_time.csv")
    
    print("number_of_rows_test")
    rows_number = [1000, 2000, 4000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    test_number_of_rows(rows_number, number_of_times=5, column_name="nn_linkage_clustering", csv_name="../results/letter_rows_time.csv")
    return 0

def test_number_of_rows(rows_number, dataset_name: str = "../datasets/real-world/letter.arff", number_of_times=5, algorithm="euclidean", formula="single", column_name="nn_linkage_clustering", csv_name="../results/letter_rows_time.csv"):
    data, number_of_clusters = get_dataset(dataset_name)
    data = data.drop(['ground_truth'], axis=1)
    results = []
    for row_number in rows_number:
        dataset = data.sample(row_number, axis=0, ignore_index=True)
        print(f"rows_size: {row_number}.")
        time_run = 0.0
        for _ in range(number_of_times):
            X_data = dataset
            
            nn_chain_linkage_alg = NNChainLinkage(algorithm, formula)
            start = time.time()
            linkage = nn_chain_linkage_alg.fit_predict(X_data.to_numpy())
            clusters = get_clusters(linkage, len(X_data), number_of_clusters)
            data[column_name] = np.nan
            for index, cluster in enumerate(clusters):
                for point in cluster:
                    data.at[point, column_name] = index
            end = time.time()
            time_run += end - start

        time_run /= number_of_times
        result = {}
        result["dataset_name"] = os.path.basename(dataset_name)
        result["rows"] = row_number
        result[column_name+"_time"] = time_run
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv(csv_name, index=False)

def test_number_of_features(columns_number, dataset_name: str = "../datasets/real-world/arrhythmia.arff", number_of_times=5, algorithm="euclidean", formula="single", column_name="nn_linkage_clustering", csv_name="../results/arrhythmia_features_time.csv"):
    data, number_of_clusters = get_dataset(dataset_name)
    data = data.drop(['ground_truth'], axis=1)
    results = []
    for column_number in columns_number:
        dataset = data.sample(column_number, axis=1)
        print(f"column_size: {column_number}.")
        time_run = 0.0
        for _ in range(number_of_times):
            X_data = dataset
            
            nn_chain_linkage_alg = NNChainLinkage(algorithm, formula)
            start = time.time()
            linkage = nn_chain_linkage_alg.fit_predict(X_data.to_numpy())
            clusters = get_clusters(linkage, len(X_data), number_of_clusters)
            data[column_name] = np.nan
            for index, cluster in enumerate(clusters):
                for point in cluster:
                    data.at[point, column_name] = index
            end = time.time()
            time_run += end - start

        time_run /= number_of_times
        result = {}
        result["dataset_name"] = os.path.basename(dataset_name)
        result["columns"] = column_number
        result[column_name+"_time"] = time_run
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv(csv_name, index=False)

def test_rand_score_time(folder: str, csv_name: str, algorithm:str, formula:str, number_of_times: int) -> None:
    datasets, number_of_clusters = get_datasets(Path(folder).glob("*"))

    print("test_agglomerative_clustering")
    df = test_agglomerative_clustering(datasets, number_of_clusters, "agglomerative_clustering",algorithm, formula, number_of_times)
    print("test_nn_linkage_clustering")
    df_nn_linkage = test_nn_linkage_clustering(datasets, number_of_clusters, "nn_linkage_clustering",algorithm, formula, number_of_times)
    df["dataset_name2"] = df_nn_linkage["dataset_name"]
    df["nn_linkage_clustering_rand_score"] = df_nn_linkage["nn_linkage_clustering_rand_score"]
    df["nn_linkage_clustering_time"] = df_nn_linkage["nn_linkage_clustering_time"]

    df.to_csv(csv_name, index=False)

def test_nn_linkage_clustering(datasets: Dict[str, pd.DataFrame], number_of_clusters: Dict[str, int], column_name: str, algorithm:str, formula:str, number_of_times: int) -> pd.DataFrame:
    results = []
    dataset_size = len(datasets)
    for index, dataset in enumerate(datasets):
        print(dataset)
        if index%5 == 0:
            print(f"Dataset: {index+1}/{dataset_size}.")
        time_run = 0.0
        score = 0.0
        for _ in range(number_of_times):
            data = datasets[dataset]
            Y_data = data["ground_truth"]
            X_data = data.drop(['ground_truth'], axis=1)
            nn_chain_linkage_alg = NNChainLinkage(algorithm, formula)
            start = time.time()
            linkage = nn_chain_linkage_alg.fit_predict(X_data.to_numpy())
            clusters = get_clusters(linkage, len(X_data), number_of_clusters[dataset])
            data[column_name] = np.nan
            for index, cluster in enumerate(clusters):
                for point in cluster:
                    data.at[point, column_name] = index
            end = time.time()
            score += metrics.rand_score( labels_true=Y_data, labels_pred=data[column_name])
            time_run += end - start

        score /= number_of_times
        time_run /= number_of_times
        result = {}
        result["dataset_name"] = dataset
        result[column_name+"_rand_score"] = score
        result[column_name+"_time"] = time_run

        results.append(result)
    df = pd.DataFrame(results)
    return df

def test_agglomerative_clustering(datasets: Dict[str, pd.DataFrame], number_of_clusters: Dict[str, int], column_name: str, algorithm:str, formula:str, number_of_times: int) -> pd.DataFrame:
    results = []
    dataset_size = len(datasets)
    for index, dataset in enumerate(datasets):
        print(dataset)
        if index%5 == 0:
            print(f"Dataset: {index+1}/{dataset_size}.")
        time_run = 0.0
        score = 0.0
        for _ in range(number_of_times):
            hierarchical_cluster = AgglomerativeClustering(n_clusters=number_of_clusters[dataset], affinity=algorithm, linkage=formula)
            data = datasets[dataset]
            Y_data = data["ground_truth"]
            X_data = data.drop(['ground_truth'], axis=1)
            start = time.time()
            labels = hierarchical_cluster.fit_predict(X_data)
            end = time.time()
            score += metrics.rand_score( labels_true=Y_data, labels_pred=labels)
            time_run += end - start
        
        score /= number_of_times
        time_run /= number_of_times
        result = {}
        result["dataset_name"] = dataset
        result[column_name+"_rand_score"] = score
        result[column_name+"_time"] = time_run
        
        results.append(result)
    df = pd.DataFrame(results)
    return df

def dev_test(dataset: str = "../datasets/artificial/target.arff") -> None: #ds3c3sc6 # artificial/target # real-world/balance-scale.arff
    data, number_of_clusters = get_dataset(dataset)
    Y_data = data["ground_truth"]
    X_data = data.drop(['ground_truth'], axis=1)

    hierarchical_cluster = AgglomerativeClustering(n_clusters=number_of_clusters, affinity='euclidean', linkage='single')
    labels = hierarchical_cluster.fit_predict(X_data)
    data["AgglomerativeClustering"] = labels
    rand_score = metrics.rand_score( labels_true=Y_data, labels_pred=labels)
    print('rand_score', rand_score)
    start = time.time()
    nn_chain_linkage_alg = NNChainLinkage()
    linkage = nn_chain_linkage_alg.fit_predict(X_data.to_numpy())
    clusters = get_clusters(linkage, len(X_data), number_of_clusters)
    end = time.time()
    print("time", end-start)
    data["NNChainLinkageClustering"] = np.nan
    start = time.time()
    for index, cluster in enumerate(clusters):
        for point in cluster:
            data.at[point, "NNChainLinkageClustering"] = index
    end = time.time()
    print("time", end-start)
    data.to_csv("test.csv", index=False)
    rand_score = metrics.rand_score( labels_true=Y_data, labels_pred=data["NNChainLinkageClustering"])
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
                         color='NNChainLinkageClustering',
                         title=f"my {dataset}")
    fig.show()
