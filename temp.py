def unique_col_values(df):
    for column in df:
        print("{} | {} | {}".format(
            df[column].name, len(df[column].unique()), df[column].dtype
        ))
unique_col_values(vehicles)
```

```
from sklearn.cluster import KMeans
def kmeans_cluster(df, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=1)
    clusters = model.fit_predict(df)
    cluster_results = df.copy()
    cluster_results['Cluster'] = clusters
    return cluster_results

def summarize_clustering(results):
    cluster_size = results.groupby(['Cluster']).size().reset_index()
    cluster_size.columns = ['Cluster', 'Count']
    cluster_means = results.groupby(['Cluster'], as_index=False).mean()
    cluster_summary = pd.merge(cluster_size, cluster_means, on='Cluster')
    return cluster_summary

import matplotlib.pyplot as plt
import seaborn as sns #CHANGE TO PLOTLY
sns.heatmap(cluster_summary[cluster_columns].transpose(), annot=True)

cluster_results['Cluster Name'] = ''
cluster_results['Cluster Name'][cluster_results['Cluster']==0] = 'Midsized Balanced'
cluster_results['Cluster Name'][cluster_results['Cluster']==1] = 'Large Inefficient'
cluster_results['Cluster Name'][cluster_results['Cluster']==2] = 'Large Moderately Efficient'
cluster_results['Cluster Name'][cluster_results['Cluster']==3] = 'Small Very Efficient'
vehicles = vehicles.reset_index().drop('index', axis=1)
vehicles['Cluster Name'] = cluster_results['Cluster Name']

def pivot_count(df, rows, columns, calc_field):
    df_pivot = df.pivot_table(values=calc_field, 
                              index=rows, 
                              columns=columns, 
                              aggfunc=np.size
                             ).dropna(axis=0, how='all')
    return df_pivot

# GET A GOOD PAIRPLOT FUNCTION

df.corr() # correlation matrix
df.cov() # covariance matrix
# CREATE OUTLIER PLOT
# CREATE TRANSFORM CONTINUOUS FUNCTION (standardize, normalize, etc.)


df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # hist for all variables

pd.set_option("display.precision", 2) # set precision

df['Churn'].value_counts(normalize=True)
