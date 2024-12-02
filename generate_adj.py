import pandas as pd
df = pd.read_csv('ML-GCN/final_stable.csv')
import pandas as pd
import numpy as np
import pickle


def compute_and_binarize_conditional_matrix(csv_path, output_path, threshold):
    """
    Compute a 70x70 conditional probability matrix from a CSV file,
    binarize it based on a threshold, and save the result as a .pkl file.
    Also prints the number of entries with 1, isolated nodes, and total nodes.

    Parameters:
    - csv_path (str): Path to the input CSV file.
    - output_path (str): Path to save the output .pkl file.
    - threshold (float): Threshold for binarizing the matrix.
    
    Returns:
    - None: Saves the binary matrix to a .pkl file.
    """
    # Load the CSV file
    # df = pd.read_csv(csv_path)
    df = pd.concat([
        pd.read_csv('ML-GCN/archive-4/final_stable_diffusion_31k.csv'),
        pd.read_csv('ML-GCN/archive-4/stable_diffusion_27k.csv'),
        pd.read_csv('ML-GCN/archive-4/artifact_presence_latent_diffusion.csv'),
        pd.read_csv('ML-GCN/archive-4/artifact_presence_giga_gan.csv'),
        pd.read_csv('ML-GCN/archive-4/artifact_presence_giga_gan_t2i_coco256.csv').dropna()
    ], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    # Drop the first and last columns to isolate the hallucination categories
    new_df = df.iloc[:, 1:-1]
    
    # Initialize the adjacency matrix
    n_categories = new_df.shape[1]  # Should be 70
    P = np.zeros((n_categories, n_categories))
    
    # Compute the conditional probabilities
    for i in range(n_categories):
        for j in range(n_categories):
            # Count the rows where both category_i and category_j are 1
            co_occurrence = np.sum((new_df.iloc[:, i] == 1) & (new_df.iloc[:, j] == 1))
            # Count the rows where category_j is 1
            total_j = np.sum(new_df.iloc[:, j] == 1)
            # Avoid division by zero
            if total_j > 0:
                P[i, j] = co_occurrence / total_j
    
    # Binarize the matrix based on the threshold
    P_binary = (P >= threshold).astype(int)
    
    # Count the number of entries with 1
    num_ones = np.sum(P_binary)
    print(f"Number of entries with 1: {num_ones}")
    
    # Count isolated nodes (rows/columns with all zeros)
    isolated_nodes = np.sum(np.all(P_binary == 0, axis=0))
    total_nodes = n_categories
    print(f"Number of isolated nodes: {isolated_nodes}")
    print(f"Total number of nodes: {total_nodes}")
    
    # Save the binary matrix to a .pkl file
    with open(output_path, 'wb') as f:
        pickle.dump(P_binary, f)
    print(f"Binary matrix saved to {output_path}")
    
    
# Example usage
compute_and_binarize_conditional_matrix("ML-GCN/final_stable.csv", "binary_matrix_0.5.pkl", threshold=0.5)
