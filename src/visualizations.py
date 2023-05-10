import pandas as pd
import plotly.express as px


arrhythmia_features_time=pd.read_csv('../results/arrhythmia_features_time.csv')
fig = px.line(arrhythmia_features_time, x="columns", y="nn_linkage_clustering_time", title='NN Linkage Clustering - effect of number of dimensions on runtime')
fig.show()

average_score_artificial_subset=pd.read_csv('../results/average_score_artificial-subset.csv')
plt=px.bar(
    data_frame = average_score_artificial_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_rand_score','nn_linkage_clustering_rand_score'],
    barmode ='group',
    title='Rand score for average metric with artificial subset',
)
plt.show()

plt=px.bar(
    data_frame = average_score_artificial_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_time','nn_linkage_clustering_time'],
    barmode ='group',
    title='Time for average metric with artificial subset',
)
plt.show()

average_score_real_world_subset=pd.read_csv('../results/average_score_real_world_subset.csv')
plt=px.bar(
    data_frame = average_score_real_world_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_rand_score','nn_linkage_clustering_rand_score'],
    barmode ='group',
    title='Rand score for average metric with real world subset',
)
plt.show()

plt=px.bar(
    data_frame = average_score_real_world_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_time','nn_linkage_clustering_time'],
    barmode ='group',
    title='Time for average metric with real world subset',
)
plt.show()

complete_rand_score_artificial_subset=pd.read_csv('../results/complete_rand_score_artificial-subset.csv')
plt=px.bar(
    data_frame = complete_rand_score_artificial_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_rand_score','nn_linkage_clustering_rand_score'],
    barmode ='group',
    title='Rand score for complete metric with artificial subset',
)
plt.show()

plt=px.bar(
    data_frame = complete_rand_score_artificial_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_time','nn_linkage_clustering_time'],
    barmode ='group',
    title='Time for complete metric with artificial subset',
)
plt.show()

complete_rand_score_real_world_subset=pd.read_csv('../results/complete_rand_score_real_world_subset.csv')
plt=px.bar(
    data_frame = complete_rand_score_real_world_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_rand_score','nn_linkage_clustering_rand_score'],
    barmode ='group',
    title='Rand score for complete metric with real world subset',
)
plt.show()

plt=px.bar(
    data_frame = complete_rand_score_real_world_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_time','nn_linkage_clustering_time'],
    barmode ='group',
    title='Time for complete metric with real world subset',
)
plt.show()

single_manhattan_rand_score_artificial_subset=pd.read_csv('../results/single_manhattan_rand_score_artificial-subset.csv')
plt=px.bar(
    data_frame = single_manhattan_rand_score_artificial_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_rand_score','nn_linkage_clustering_rand_score'],
    barmode ='group',
    title='Rand score for single metric and manhattan with artificial subset',
)
plt.show()

plt=px.bar(
    data_frame = single_manhattan_rand_score_artificial_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_time','nn_linkage_clustering_time'],
    barmode ='group',
    title='Time for single metric and manhattan with artificial subset',
)
plt.show()

single_manhattan_rand_score_real_world_subset=pd.read_csv('../results/single_manhattan_rand_score_real_world_subset.csv')
plt=px.bar(
    data_frame = single_manhattan_rand_score_real_world_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_rand_score','nn_linkage_clustering_rand_score'],
    barmode ='group',
    title='Rand score for single metric and manhattan with real world subset',
)
plt.show()

plt=px.bar(
    data_frame = single_manhattan_rand_score_real_world_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_time','nn_linkage_clustering_time'],
    barmode ='group',
    title='Time for single metric and manhattan with real world subset',
)
plt.show()

single_rand_score_artificial_subset=pd.read_csv('../results/single_rand_score_artificial-subset.csv')
plt=px.bar(
    data_frame = single_rand_score_artificial_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_rand_score','nn_linkage_clustering_rand_score'],
    barmode ='group',
    title='Rand score for single metric with artificial subset',
)
plt.show()

plt=px.bar(
    data_frame = single_rand_score_artificial_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_time','nn_linkage_clustering_time'],
    barmode ='group',
    title='Time for single metric with artificial subset',
)
plt.show()

single_rand_score_real_world_subset=pd.read_csv('../results/single_rand_score_real_world_subset.csv')
plt=px.bar(
    data_frame = single_rand_score_real_world_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_rand_score','nn_linkage_clustering_rand_score'],
    barmode ='group',
    title='Rand score for single metric with real world subset',
)
plt.show()

plt=px.bar(
    data_frame = single_rand_score_real_world_subset,
    x ='dataset_name',
    y = ['agglomerative_clustering_time','nn_linkage_clustering_time'],
    barmode ='group',
    title='Time for single metric with real world subset',
)
plt.show()