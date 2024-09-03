# Main6 Confusion matrix and multi-class ROC curves
# Plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, base_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8), dpi=150)  # Increase resolution with dpi=150
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title('Confusion Matrix', fontsize=20, weight='bold')
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'), dpi=150)
    plt.show()
    print(f"Confusion matrix plot saved to: {os.path.join(base_dir, 'confusion_matrix.png')}")

# Plot multi-class ROC curves
def plot_multiclass_roc(y_true, y_pred, base_dir, n_classes=10):
    y_true_bin = label_binarize(y_true, classes=[i for i in range(n_classes)])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(12, 10), dpi=150)  # Increase resolution with dpi=150
    colors = sns.color_palette("tab10", n_colors=n_classes)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curves for Each Class', fontsize=20, weight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'multiclass_roc_curve.png'), dpi=150)
    plt.show()
    print(f"Multiclass ROC curve plot saved to: {os.path.join(base_dir, 'multiclass_roc_curve.png')}")

# Visualize the comparison of predicted results and actual labels
def plot_predictions(y_true, y_pred_classes, base_dir):
    plt.figure(figsize=(14, 8), dpi=150)  # Increase resolution with dpi=150
    plt.scatter(range(len(y_true)), y_true, color="blue", alpha=0.6, label="True Labels", s=50)
    plt.scatter(range(len(y_pred_classes)), y_pred_classes, color="red", alpha=0.4, label="Predicted Labels", s=30)
    plt.title('True vs Predicted Labels', fontsize=20, weight='bold')
    plt.xlabel('Sample Index', fontsize=16)
    plt.ylabel('Label', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'true_vs_predicted.png'), dpi=150)
    plt.show()
    print(f"True vs Predicted labels plot saved to: {os.path.join(base_dir, 'true_vs_predicted.png')}")

# Main program: Load and evaluate the model
def main6():
    base_dir = "E:\japan"
    
    # Load preprocessed test data from disk
    X_test = np.load(os.path.join(base_dir, "X_test.npy"))
    y_test = np.load(os.path.join(base_dir, "y_test.npy"))

    # Load the trained classification model cnn_classifier_final.h5 or best_cnn_classifier.h5
    model = load_model(os.path.join(base_dir, "best_cnn_classifier.h5"))

    # Evaluate the model's performance on the test set
    print("Evaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print classification report
    class_names = [f"Class {i}" for i in range(10)]
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))

    # Plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, class_names, base_dir)

    # Plot multi-class ROC curves
    plot_multiclass_roc(y_test, y_pred, base_dir, n_classes=10)

    # Visualize the comparison of predicted results and actual labels
    plot_predictions(y_test, y_pred_classes, base_dir)

if __name__ == "__main__":
    main6()


