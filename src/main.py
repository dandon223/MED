import argparse
from tests import get_dataset, get_numerical_dataset
import json
import numpy as np
import pandas as pd
import time
from sklearn import metrics
from nn_chain_linkage import NNChainLinkage, get_clusters

def run(config_file: str):
    with open(config_file) as f:
        data = json.load(f)
        input_file = data['input_file']
        algorithm = data['algorithm']
        formula = data['formula']
        output_file = data['output_file']
        is_numerical = data['is_numerical']
        if is_numerical == "True":
            data, number_of_clusters = get_numerical_dataset(input_file)
        else:
            data, number_of_clusters = get_dataset(input_file)
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
        time_run = end - start
        score = metrics.rand_score(labels_true=Y_data, labels_pred=clusters_index)
        result = {}
        result["dataset_name"] = input_file
        result["score"] = score
        result["time"] = time_run
        df = pd.DataFrame(result, index=[0])
        df.to_csv(output_file, index=False)
        return score, time_run, clusters_index


def main():
    parser = argparse.ArgumentParser(description='MED: Automatic testing suit')

    parser.add_argument('--config_file', help='config file for algorithm', type=str)
    
    args = parser.parse_args()

    if args.config_file is not None:
        run(args.config_file)

if __name__ == "__main__":
    main()
