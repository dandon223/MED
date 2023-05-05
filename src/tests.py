from scipy.io import arff
from typing import List, Tuple
import pandas as pd
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
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

    return df, number_of_clusters

def dev_test(dataset: str = "../datasets/artificial/target.arff") -> None:
    data, number_of_clusters = get_dataset(dataset)
    print("data.shape", data.shape)
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

    print(data.head())
    data["AgglomerativeClustering"] = labels

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
                         title=f"CUREClustering {dataset}")
    fig.show()
