# Main4 data processing and split the dataset for CNN classify model
# Balance the dataset
def balance_dataset(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X.reshape(len(X), -1), y)
    return X_balanced.reshape(-1, X.shape[1], X.shape[2]), y_balanced

# Preprocess the data and split into training, validation, and test sets
def preprocess_and_split_data(freq_features, clusters):
    # Standardize frequency features
    scaler_freq = StandardScaler()
    freq_features_normalized = scaler_freq.fit_transform(freq_features.reshape(-1, freq_features.shape[-1])).reshape(freq_features.shape)
    print(f"Normalized Frequency Features Shape: {freq_features_normalized.shape}")

    # Plot histogram of labels to show distribution before balancing
    plt.figure(figsize=(8, 6))
    sns.histplot(clusters, bins=len(np.unique(clusters)), kde=False, color='skyblue')
    plt.title('Cluster Label Distribution Before Balancing', fontsize=16)
    plt.xlabel('Cluster Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True)
    plt.show()

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(freq_features_normalized, clusters, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Balance the training set
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    
    # Plot histogram of labels after balancing
    plt.figure(figsize=(8, 6))
    sns.histplot(y_train_balanced, bins=len(np.unique(y_train_balanced)), kde=False, color='orange')
    plt.title('Cluster Label Distribution After Balancing', fontsize=16)
    plt.xlabel('Cluster Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True)
    plt.show()

    # Display the shape of the datasets
    print(f"Training Set Shape: {X_train_balanced.shape}, Validation Set Shape: {X_val.shape}, Test Set Shape: {X_test.shape}")
    
    # Visualize the number of samples in each dataset split
    datasets = ['Training', 'Validation', 'Test']
    dataset_sizes = [len(X_train_balanced), len(X_val), len(X_test)]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=datasets, y=dataset_sizes, palette='viridis')
    plt.title('Dataset Split Sizes', fontsize=16)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.grid(True)
    plt.show()

    return (X_train_balanced, X_val, X_test), (y_train_balanced, y_val, y_test)

# Main program: Preprocess data and split into training, validation, and test sets
def main4():
    base_dir = "E:\japan"
    
    freq_features = np.load(os.path.join(base_dir, "freq_features.npy"))
    cluster_labels = np.load(os.path.join(base_dir, "gmm_cluster_labels_umap.npy"))

    if freq_features is None or cluster_labels is None:
        print("Failed to load frequency features or cluster labels, exiting...")
        return

    # Data preprocessing and splitting
    (X_train, X_val, X_test), (y_train, y_val, y_test) = preprocess_and_split_data(freq_features, cluster_labels)

    # Save the processed datasets
    np.save(os.path.join(base_dir, "X_train.npy"), X_train)
    np.save(os.path.join(base_dir, "X_val.npy"), X_val)
    np.save(os.path.join(base_dir, "X_test.npy"), X_test)
    np.save(os.path.join(base_dir, "y_train.npy"), y_train)
    np.save(os.path.join(base_dir, "y_val.npy"), y_val)
    np.save(os.path.join(base_dir, "y_test.npy"), y_test)
    print("Datasets have been saved to disk")

if __name__ == "__main__":
    main4()

