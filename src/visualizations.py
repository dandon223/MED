import pandas as pd
import plotly.express as px

RESULTS_FOLDER = '../results/'
DATASET_NAME = 'dataset_name'
AGGLOMERATIVE_CLUSTERING_RAND_SCORE = 'agglomerative_clustering_rand_score'
NN_LINKAGE_CLUSTERING_RAND_SCORE = 'nn_linkage_clustering_rand_score'
AGGLOMERATIVE_CLUSTERING_TIME = 'agglomerative_clustering_time'
NN_LINKAGE_CLUSTERING_TIME = 'nn_linkage_clustering_time'
GROUP = 'group'

arrhythmia_features_time=pd.read_csv(RESULTS_FOLDER + 'arrhythmia_features_time.csv')
fig = px.line(arrhythmia_features_time, x="columns", y=NN_LINKAGE_CLUSTERING_TIME, title='NN Linkage Clustering - effect of number of dimensions on runtime')
fig.show()

letter_rows_time=pd.read_csv(RESULTS_FOLDER + 'letter_rows_time.csv')
fig = px.line(letter_rows_time, x="rows", y=NN_LINKAGE_CLUSTERING_TIME, title='NN Linkage Clustering - effect of number of rows on runtime')
fig.show()

average_score_artificial_subset=pd.read_csv(RESULTS_FOLDER + 'average_score_artificial-subset.csv')
plt=px.bar(
    data_frame = average_score_artificial_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_RAND_SCORE,NN_LINKAGE_CLUSTERING_RAND_SCORE],
    barmode =GROUP,
    title='Rand score for average metric with artificial subset',
)
plt.show()

plt=px.bar(
    data_frame = average_score_artificial_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_TIME,NN_LINKAGE_CLUSTERING_TIME],
    barmode =GROUP,
    title='Time for average metric with artificial subset',
)
plt.show()

average_score_real_world_subset=pd.read_csv(RESULTS_FOLDER + 'average_score_real_world_subset.csv')
plt=px.bar(
    data_frame = average_score_real_world_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_RAND_SCORE,NN_LINKAGE_CLUSTERING_RAND_SCORE],
    barmode =GROUP,
    title='Rand score for average metric with real world subset',
)
plt.show()

plt=px.bar(
    data_frame = average_score_real_world_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_TIME,NN_LINKAGE_CLUSTERING_TIME],
    barmode =GROUP,
    title='Time for average metric with real world subset',
)
plt.show()

complete_rand_score_artificial_subset=pd.read_csv(RESULTS_FOLDER + 'complete_rand_score_artificial-subset.csv')
plt=px.bar(
    data_frame = complete_rand_score_artificial_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_RAND_SCORE,NN_LINKAGE_CLUSTERING_RAND_SCORE],
    barmode =GROUP,
    title='Rand score for complete metric with artificial subset',
)
plt.show()

plt=px.bar(
    data_frame = complete_rand_score_artificial_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_TIME,NN_LINKAGE_CLUSTERING_TIME],
    barmode =GROUP,
    title='Time for complete metric with artificial subset',
)
plt.show()

complete_rand_score_real_world_subset=pd.read_csv(RESULTS_FOLDER + 'complete_rand_score_real_world_subset.csv')
plt=px.bar(
    data_frame = complete_rand_score_real_world_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_RAND_SCORE,NN_LINKAGE_CLUSTERING_RAND_SCORE],
    barmode =GROUP,
    title='Rand score for complete metric with real world subset',
)
plt.show()

plt=px.bar(
    data_frame = complete_rand_score_real_world_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_TIME,NN_LINKAGE_CLUSTERING_TIME],
    barmode =GROUP,
    title='Time for complete metric with real world subset',
)
plt.show()

single_manhattan_rand_score_artificial_subset=pd.read_csv(RESULTS_FOLDER + 'single_manhattan_rand_score_artificial-subset.csv')
plt=px.bar(
    data_frame = single_manhattan_rand_score_artificial_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_RAND_SCORE,NN_LINKAGE_CLUSTERING_RAND_SCORE],
    barmode =GROUP,
    title='Rand score for single metric and manhattan with artificial subset',
)
plt.show()

plt=px.bar(
    data_frame = single_manhattan_rand_score_artificial_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_TIME,NN_LINKAGE_CLUSTERING_TIME],
    barmode =GROUP,
    title='Time for single metric and manhattan with artificial subset',
)
plt.show()

single_manhattan_rand_score_real_world_subset=pd.read_csv(RESULTS_FOLDER + 'single_manhattan_rand_score_real_world_subset.csv')
plt=px.bar(
    data_frame = single_manhattan_rand_score_real_world_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_RAND_SCORE,NN_LINKAGE_CLUSTERING_RAND_SCORE],
    barmode =GROUP,
    title='Rand score for single metric and manhattan with real world subset',
)
plt.show()

plt=px.bar(
    data_frame = single_manhattan_rand_score_real_world_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_TIME,NN_LINKAGE_CLUSTERING_TIME],
    barmode =GROUP,
    title='Time for single metric and manhattan with real world subset',
)
plt.show()

single_rand_score_artificial_subset=pd.read_csv(RESULTS_FOLDER + 'single_rand_score_artificial-subset.csv')
plt=px.bar(
    data_frame = single_rand_score_artificial_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_RAND_SCORE,NN_LINKAGE_CLUSTERING_RAND_SCORE],
    barmode =GROUP,
    title='Rand score for single metric with artificial subset',
)
plt.show()

plt=px.bar(
    data_frame = single_rand_score_artificial_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_TIME,NN_LINKAGE_CLUSTERING_TIME],
    barmode =GROUP,
    title='Time for single metric with artificial subset',
)
plt.show()

single_rand_score_real_world_subset=pd.read_csv(RESULTS_FOLDER + 'single_rand_score_real_world_subset.csv')
plt=px.bar(
    data_frame = single_rand_score_real_world_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_RAND_SCORE,NN_LINKAGE_CLUSTERING_RAND_SCORE],
    barmode =GROUP,
    title='Rand score for single metric with real world subset',
)
plt.show()

plt=px.bar(
    data_frame = single_rand_score_real_world_subset,
    x =DATASET_NAME,
    y = [AGGLOMERATIVE_CLUSTERING_TIME,NN_LINKAGE_CLUSTERING_TIME],
    barmode =GROUP,
    title='Time for single metric with real world subset',
)
plt.show()