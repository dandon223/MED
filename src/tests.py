import argparse
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

NN_LINKAGE_CLUSTERING = 'nn_linkage_clustering'
NN_LINKAGE_CLUSTERING_RAND_SCORE = NN_LINKAGE_CLUSTERING + '_rand_score'
NN_LINKAGE_CLUSTERING_TIME = NN_LINKAGE_CLUSTERING + '_time'

AGGLOMERATIVE_CLUSTERING = 'agglomerative_clustering'
AGGLOMERATIVE_CLUSTERING_RAND_SCORE = AGGLOMERATIVE_CLUSTERING + '_rand_score'
AGGLOMERATIVE_CLUSTERING_TIME = AGGLOMERATIVE_CLUSTERING + '_time'

def get_other_dataset(file: str)  -> Tuple[pd.DataFrame, int]:

    dataset = pd.read_csv(file, header=None, dtype="string")
    number_of_clusters = dataset.iloc[:, -1].nunique()
    dataset = dataset.rename(columns={dataset.columns[-1]: "ground_truth"})
    return dataset, number_of_clusters

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

    artificial = '../datasets/artificial-subset'
    real_world = '../datasets/real-world-subset'
    results_folder = '../results/'
    print("single_test_rand_score_artificial")
    test_rand_score_time(artificial, results_folder + "single_rand_score_artificial-subset.csv", "euclidean", "single", 1)
    print("single_test_rand_score_real_world_subset")
    test_rand_score_time(real_world, results_folder + "single_rand_score_real_world_subset.csv", "euclidean", "single", 1)

    print("complete_test_rand_score_artificial")
    test_rand_score_time(artificial, results_folder + "complete_rand_score_artificial-subset.csv", "euclidean", "complete", 1)
    print("complete_test_rand_score_real_world_subset")
    test_rand_score_time(real_world, results_folder + "complete_rand_score_real_world_subset.csv", "euclidean", "complete", 1)

    print("average_test_rand_score_artificial")
    test_rand_score_time(artificial, results_folder + "average_score_artificial-subset.csv", "euclidean", "average", 1)
    print("average_test_rand_score_real_world_subset")
    test_rand_score_time(real_world, results_folder + "average_score_real_world_subset.csv", "euclidean", "average", 1)

    print("single_manhattan_test_rand_score_artificial")
    test_rand_score_time(artificial, results_folder + "single_manhattan_rand_score_artificial-subset.csv", "manhattan", "single", 1)
    print("single_manhattan_test_rand_score_real_world_subset")
    test_rand_score_time(real_world, results_folder + "single_manhattan_rand_score_real_world_subset.csv", "manhattan", "single", 1)

    print("number_of_features_test")
    columns_number = [2, 4, 8, 16, 32, 64, 128, 256]
    test_number_of_features(columns_number, results_folder + "arrhythmia_features_time.csv", number_of_times=5)
    
    print("number_of_rows_test")
    rows_number = [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    test_number_of_rows(rows_number, results_folder + "letter_rows_time.csv", number_of_times=5)

    print("type_of_algorithm_test")
    type_of_algorithm_test(artificial, results_folder + "type_of_algorithm_time.csv", 5, "single")

    print("numerical_data_test")
    numerical_data_test('../datasets/agaricus-lepiota.data', results_folder + "numerical_data_test.csv", 5)
    return 0

def numerical_data_test(file: str, csv_name:str, number_of_times: int):

    dataset, number_of_clusters = get_other_dataset(file)
    formulas=['single', 'complete']
    result = {}
    for formula in formulas:
        time_run = 0.0
        score = 0.0
        for _ in range(number_of_times):
            score_temp, time_run_temp, _ = run_nn_linkage_clustering(dataset, number_of_clusters, 'gower', formula)
            score += score_temp
            time_run += time_run_temp
        score /= number_of_times
        time_run /= number_of_times

        result["dataset_name"] = os.path.basename(file)
        result[NN_LINKAGE_CLUSTERING_RAND_SCORE+ formula] = score
        result[NN_LINKAGE_CLUSTERING_TIME+ formula] = time_run
    
    df = pd.DataFrame(result, index=[0])
    df.to_csv(csv_name, index=False)

def type_of_algorithm_test(folder: str, csv_name:str, number_of_times: int, formula:str):

    results = []
    subset=["R15.arff", "blobs.arff",  "target.arff"]
    algorithms=["manhattan", "euclidean"]
    datasets, number_of_clusters = get_datasets(Path(folder).glob("*"))
    for dataset in subset:
        result = {}
        for algorithm in algorithms:
            time_run = 0.0
            score = 0.0
            for _ in range(number_of_times):
                score_temp, time_run_temp, _ = run_nn_linkage_clustering(datasets[dataset], number_of_clusters[dataset], algorithm, formula)
                score += score_temp
                time_run += time_run_temp
            score /= number_of_times
            time_run /= number_of_times
            
            result["dataset_name"] = dataset
            result[NN_LINKAGE_CLUSTERING_RAND_SCORE+ algorithm] = score
            result[NN_LINKAGE_CLUSTERING_TIME+ algorithm] = time_run

        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv(csv_name, index=False)

def test_number_of_rows(rows_number: int,csv_name:str, dataset_name: str = "../datasets/letter.arff", number_of_times: int=5, algorithm: str="euclidean", formula: str="single"):

    data, number_of_clusters = get_dataset(dataset_name)
    results = []
    for row_number in rows_number:
        dataset = data.sample(row_number, axis=0, ignore_index=True)
        print(f"rows_size: {row_number}.")
        time_run = 0.0
        for _ in range(number_of_times):
            _, time_run_temp, _ = run_nn_linkage_clustering(dataset, number_of_clusters, algorithm, formula)
            print("time:", time_run_temp)
            time_run += time_run_temp

        time_run /= number_of_times
        result = {}
        result["dataset_name"] = os.path.basename(dataset_name)
        result["rows"] = row_number
        result[NN_LINKAGE_CLUSTERING_TIME] = time_run
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv(csv_name, index=False)

def test_number_of_features(columns_number, csv_name:str, dataset_name: str = "../datasets/arrhythmia.arff", number_of_times=5, algorithm="euclidean", formula="single"):

    data, number_of_clusters = get_dataset(dataset_name)
    Y = data['ground_truth']
    data = data.drop(['ground_truth'], axis=1)
    results = []
    for column_number in columns_number:
        dataset = data.sample(column_number, axis=1)
        dataset['ground_truth'] = Y
        print(f"column_size: {column_number}.")
        time_run = 0.0
        for _ in range(number_of_times):
            _, time_run_temp, _ = run_nn_linkage_clustering(dataset, number_of_clusters, algorithm, formula)
            print("time:", time_run_temp)
            time_run += time_run_temp

        time_run /= number_of_times
        result = {}
        result["dataset_name"] = os.path.basename(dataset_name)
        result["columns"] = column_number
        result[NN_LINKAGE_CLUSTERING_TIME] = time_run
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv(csv_name, index=False)

def test_rand_score_time(folder: str, csv_name: str, algorithm:str, formula:str, number_of_times: int) -> None:

    datasets, number_of_clusters = get_datasets(Path(folder).glob("*"))

    print("test_agglomerative_clustering")
    df = test_agglomerative_clustering(datasets, number_of_clusters, algorithm, formula, number_of_times)
    print("test_nn_linkage_clustering")
    df_nn_linkage = test_nn_linkage_clustering(datasets, number_of_clusters, algorithm, formula, number_of_times)
    df["dataset_name2"] = df_nn_linkage["dataset_name"]
    df[NN_LINKAGE_CLUSTERING_RAND_SCORE] = df_nn_linkage[NN_LINKAGE_CLUSTERING_RAND_SCORE]
    df[NN_LINKAGE_CLUSTERING_TIME] = df_nn_linkage[NN_LINKAGE_CLUSTERING_TIME]

    df.to_csv(csv_name, index=False)

def run_nn_linkage_clustering(data: pd.DataFrame, number_of_clusters: int, algorithm: str, formula: str ) -> Tuple[float, float, pd.Series]:

    Y_data = data["ground_truth"]
    X_data = data.drop(['ground_truth'], axis=1)
    nn_chain_linkage_alg = NNChainLinkage(algorithm, formula)
    start = time.time()
    linkage = nn_chain_linkage_alg.fit_predict(X_data.to_numpy())
    clusters = get_clusters(linkage, len(X_data), number_of_clusters)
    clusters_index = [np.nan for _ in range(len(data.index))]
    for index, cluster in enumerate(clusters):
        for point in cluster:
            clusters_index[point] = index
    end = time.time()
    score = metrics.rand_score( labels_true=Y_data, labels_pred=clusters_index)
    time_run = end - start
    return score, time_run, clusters_index

def test_nn_linkage_clustering(datasets: Dict[str, pd.DataFrame], number_of_clusters: Dict[str, int], algorithm:str, formula:str, number_of_times: int) -> pd.DataFrame:

    results = []
    dataset_size = len(datasets)
    for index, dataset in enumerate(datasets):
        if index%5 == 0:
            print(f"Dataset: {index+1}/{dataset_size}.")
        time_run = 0.0
        score = 0.0
        data = datasets[dataset]
        for _ in range(number_of_times):
            score_temp, time_run_temp, _ = run_nn_linkage_clustering(data, number_of_clusters[dataset], algorithm, formula)
            score += score_temp
            time_run += time_run_temp

        score /= number_of_times
        time_run /= number_of_times
        result = {}
        result["dataset_name"] = dataset
        result[NN_LINKAGE_CLUSTERING_RAND_SCORE] = score
        result[NN_LINKAGE_CLUSTERING_TIME] = time_run

        results.append(result)
    df = pd.DataFrame(results)
    return df

def run_agglomerative_clustering(data: pd.DataFrame, number_of_clusters: int, algorithm: str, formula: str ) -> Tuple[float, float, pd.Series]:

    hierarchical_cluster = AgglomerativeClustering(n_clusters=number_of_clusters, affinity=algorithm, linkage=formula)
    Y_data = data["ground_truth"]
    X_data = data.drop(['ground_truth'], axis=1)
    start = time.time()
    labels = hierarchical_cluster.fit_predict(X_data)
    end = time.time()
    score = metrics.rand_score( labels_true=Y_data, labels_pred=labels)
    time_run = end - start
    return score, time_run, labels

def test_agglomerative_clustering(datasets: Dict[str, pd.DataFrame], number_of_clusters: Dict[str, int], algorithm:str, formula:str, number_of_times: int) -> pd.DataFrame:

    results = []
    dataset_size = len(datasets)
    for index, dataset in enumerate(datasets):
        if index%5 == 0:
            print(f"Dataset: {index+1}/{dataset_size}.")
        time_run = 0.0
        score = 0.0
        data = datasets[dataset]
        for _ in range(number_of_times):
            score_temp, time_run_temp, _ = run_agglomerative_clustering(data, number_of_clusters[dataset], algorithm, formula)
            score += score_temp
            time_run += time_run_temp
        
        score /= number_of_times
        time_run /= number_of_times
        result = {}
        result["dataset_name"] = dataset
        result[AGGLOMERATIVE_CLUSTERING_RAND_SCORE] = score
        result[AGGLOMERATIVE_CLUSTERING_TIME] = time_run
        
        results.append(result)
    df = pd.DataFrame(results)
    return df

def dev_test(dataset: str = "../datasets/artificial-subset/R15.arff") -> None: # artificial/target # real-world/balance-scale.arff

    data, number_of_clusters = get_dataset(dataset)
    score, time_run, agglomerative_clustering_labels = run_agglomerative_clustering(data, number_of_clusters, 'euclidean', 'single')

    print('rand_score', score)
    print('time', time_run)
    
    score, time_run, nn_chainLinkage_clustering_labels = run_nn_linkage_clustering(data, number_of_clusters, 'euclidean', 'single')
    print('rand_score', score)
    print('time', time_run)
    
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
                         color=agglomerative_clustering_labels,
                         title=f"AgglomerativeClustering {dataset}")
    fig.show()

    fig = px.scatter(data,
                         x='x',
                         y='y',
                         color=nn_chainLinkage_clustering_labels,
                         title=f"my {dataset}")
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MED: Automatic testing suit')

    parser.add_argument('--dev_test', help='run test suit for development', type=str)
    parser.add_argument('--test', help='run test suit', type=str)
    
    args = parser.parse_args()

    if args.dev_test is not None:
        dev_test()
    elif args.test is not None:
        test()