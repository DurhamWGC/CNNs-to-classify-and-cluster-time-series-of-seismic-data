# Main3 Kmeans and GMM Clustering with CNN encoder
# Extract features using the encoder and perform clustering
def encoder_feature_extraction_and_clustering(X, encoder, n_clusters=10, pca_components=2):
    encoded_features = encoder.predict(X)
    print(f"Extracted encoder features shape: {encoded_features.shape}")

    # Flatten the features to fit the input format of clustering algorithms
    flattened_features = encoded_features.reshape(encoded_features.shape[0], -1)
    scaler = StandardScaler()
    flattened_features = scaler.fit_transform(flattened_features)

    # PCA dimensionality reduction
    pca = PCA(n_components=pca_components)
    pca_reduced_features = pca.fit_transform(flattened_features)
    print(f"PCA Reduced features shape: {pca_reduced_features.shape}")
    
    # UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=pca_components, random_state=42)
    umap_reduced_features = reducer.fit_transform(flattened_features)
    print(f"UMAP Reduced features shape: {umap_reduced_features.shape}")

    return flattened_features, pca_reduced_features, umap_reduced_features

# Evaluate clustering performance
def evaluate_clustering_performance(features, labels):
    score = silhouette_score(features, labels)
    print(f'Silhouette Score: {score:.4f}')
    return score

# Perform clustering using KMeans
def kmeans_clustering_and_plot(reduced_features, flattened_features, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(flattened_features)

    # Evaluate clustering performance
    evaluate_clustering_performance(flattened_features, cluster_labels)

    # Visualize clustering results
    plt.figure(figsize=(14, 10))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=cluster_labels, palette='viridis', s=100, alpha=0.7, edgecolor='k')
    plt.title("KMeans Clustering Results", fontsize=20, fontweight='bold')
    plt.xlabel("Component 1", fontsize=16)
    plt.ylabel("Component 2", fontsize=16)
    plt.legend(title="Cluster Labels", loc='upper right', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return cluster_labels

# Perform clustering using GMM
def gmm_clustering_and_plot(reduced_features, flattened_features, n_clusters=10):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    cluster_labels = gmm.fit_predict(flattened_features)

    # Evaluate clustering performance
    evaluate_clustering_performance(flattened_features, cluster_labels)

    # Visualize clustering results
    plt.figure(figsize=(14, 10))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=cluster_labels, palette='viridis', s=100, alpha=0.7, edgecolor='k')
    plt.title("GMM Clustering Results", fontsize=20, fontweight='bold')
    plt.xlabel("Component 1", fontsize=16)
    plt.ylabel("Component 2", fontsize=16)
    plt.legend(title="Cluster Labels", loc='upper right', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return cluster_labels

# Main program: Load the saved encoder model and perform clustering
def main3(n_clusters=10):
    base_dir = "E:\japan"
    
    # Load data from disk
    try:
        freq_features = np.load(os.path.join(base_dir, "freq_features.npy"))
        if freq_features is None or len(freq_features) == 0:
            raise ValueError("Frequency features data is empty.")
    except Exception as e:
        print(f"Error loading frequency features: {e}")
        return

    # Data standardization
    scaler = StandardScaler()
    freq_features = scaler.fit_transform(freq_features.reshape(-1, freq_features.shape[-1])).reshape(freq_features.shape)

    # Load the saved encoder model
    encoder = tf.keras.models.load_model(os.path.join(base_dir, "encoder_model.h5"))
    print("Encoder model loaded.")

    # Extract features using the encoder and perform dimensionality reduction
    flattened_features, pca_reduced_features, umap_reduced_features = encoder_feature_extraction_and_clustering(freq_features, encoder, n_clusters, pca_components=2)

    # Perform KMeans clustering and save the results (PCA)
    kmeans_cluster_labels_pca = kmeans_clustering_and_plot(pca_reduced_features, flattened_features, n_clusters)
    np.save(os.path.join(base_dir, "kmeans_cluster_labels_pca.npy"), kmeans_cluster_labels_pca)
    print("KMeans cluster labels (PCA) saved to: kmeans_cluster_labels_pca.npy")

    # Perform KMeans clustering and save the results (UMAP)
    kmeans_cluster_labels_umap = kmeans_clustering_and_plot(umap_reduced_features, flattened_features, n_clusters)
    np.save(os.path.join(base_dir, "kmeans_cluster_labels_umap.npy"), kmeans_cluster_labels_umap)
    print("KMeans cluster labels (UMAP) saved to: kmeans_cluster_labels_umap.npy")

    # Perform GMM clustering and save the results (PCA)
    gmm_cluster_labels_pca = gmm_clustering_and_plot(pca_reduced_features, flattened_features, n_clusters)
    np.save(os.path.join(base_dir, "gmm_cluster_labels_pca.npy"), gmm_cluster_labels_pca)
    print("GMM cluster labels (PCA) saved to: gmm_cluster_labels_pca.npy")

    # Perform GMM clustering and save the results (UMAP)
    gmm_cluster_labels_umap = gmm_clustering_and_plot(umap_reduced_features, flattened_features, n_clusters)
    np.save(os.path.join(base_dir, "gmm_cluster_labels_umap.npy"), gmm_cluster_labels_umap)
    print("GMM cluster labels (UMAP) saved to: gmm_cluster_labels_umap.npy")

if __name__ == "__main__":
    main3()
