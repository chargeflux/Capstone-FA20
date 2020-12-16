import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def calculate_distance(known_program_names, centroid_coordinate, unk_coordinate):
    distances = []
    for centroid in centroid_coordinate:
        distances.append(distance.euclidean(centroid, unk_coordinate))
    return known_program_names[np.argmax(distances)]

def get_PCA_result(X, Y):
    pca = PCA(n_components=5, random_state=1)
    pca_components = pca.fit_transform(X)
    pca_result = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

    # Keep top 3 principal components
    data_pca = pd.concat([Y, pca_result[['PC1', 'PC2', 'PC3']]], axis=1)
    pca_known = data_pca.iloc[:-1]
    pca_unknown = data_pca.iloc[-1:].values.tolist()
    assert len(pca_unknown) == 1

    # Get coordinates for centroids of known programs
    mean_df = pca_known.groupby('program_name').mean().reset_index()
    known_program_names = mean_df.program_name.values.tolist()
    pca_known_centroid = mean_df.loc[:,mean_df.columns != 'program_name'].values.tolist()

    print("Detect outlier via PCA")
    outlier = None
    for unk_program in pca_unknown:
        outlier = calculate_distance(known_program_names, pca_known_centroid, unk_program[1:])

    return outlier

def get_TSNE_result(X, Y):
    tsne = TSNE(n_components=3, learning_rate=20, perplexity=6, random_state=1)
    X_embedded = tsne.fit_transform(X)
    tsne_result = pd.DataFrame(data=X_embedded, columns=['PC1', 'PC2', 'PC3'])
    data_tsne = pd.concat([Y, tsne_result], axis=1)
    tsne_known = data_tsne.iloc[:-1]
    tsne_unknown = data_tsne.iloc[-1:].values.tolist()
    assert len(tsne_unknown) == 1

    # Get coordinates for centroids of known programs
    mean_df = tsne_known.groupby('program_name').mean().reset_index()
    known_program_names = mean_df.program_name.values.tolist()
    tsne_known_centroid = mean_df.loc[:, mean_df.columns != 'program_name'].values.tolist()

    print("Detect outlier via T-SNE")
    outlier=None
    for unk_program in tsne_unknown:
        outlier = calculate_distance(known_program_names, tsne_known_centroid, unk_program[1:])
    
    return outlier

def detect_outliers(data):
    X = data.loc[:, data.columns != 'program_name']
    Y = data[['program_name']]

    pca_outlier = get_PCA_result(X, Y)
    tsne_outlier = get_TSNE_result(X, Y)
    if pca_outlier != tsne_outlier:
        return None
    else:
        return pca_outlier