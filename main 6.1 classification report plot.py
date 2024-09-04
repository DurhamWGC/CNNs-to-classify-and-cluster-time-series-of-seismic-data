# main 6.1 classification report plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm  # For progress bar display
from tensorflow.keras.models import load_model  # For loading Keras model
import os

def plot_classification_report_with_progress(y_true, y_pred, class_names, base_dir):
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Set the total number of progress steps
    steps = 4
    with tqdm(total=steps, desc="Generating classification report plot") as pbar:
        
        # 1. Initialize the plot
        plt.figure(figsize=(12, 8), dpi=150)
        pbar.update(1)  # Update progress bar
        
        # 2. Plot the heatmap
        sns.heatmap(
            report_df.iloc[:-1, :-1].T, 
            annot=True, 
            cmap="coolwarm",  # Use a light color palette
            cbar=False, 
            fmt=".2f",
            annot_kws={"size": 12, "weight": "bold"},  # Set font size and bold text
            linewidths=0.5  # Increase line width for better contrast
        )
        pbar.update(1)  # Update progress bar
        
        # 3. Set title and labels
        plt.title('Classification Report', fontsize=20, weight='bold')
        plt.xticks(fontsize=12, weight='bold')
        plt.yticks(fontsize=12, weight='bold', rotation=0)
        plt.tight_layout()
        pbar.update(1)  # Update progress bar
        
        # 4. Save the plot
        plt.savefig(os.path.join(base_dir, 'classification_report.png'), dpi=150)
        plt.show()
        pbar.update(1)  # Update progress bar

    print(f"Classification report plot saved at: {os.path.join(base_dir, 'classification_report.png')}")

# Main program
def main():
    base_dir = "E:\japan"
    
    # Load the model
    model = load_model(os.path.join(base_dir, "best_cnn_classifier.h5"))

    # Load test data
    X_test = np.load(os.path.join(base_dir, "X_test.npy"))
    y_test = np.load(os.path.join(base_dir, "y_test.npy"))

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Class name list
    class_names = [f"Class {i}" for i in range(10)]

    # Call function to plot the classification report
    plot_classification_report_with_progress(y_test, y_pred_classes, class_names, base_dir)

if __name__ == "__main__":
    main()
