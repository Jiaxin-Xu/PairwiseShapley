from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
from tabulate import tabulate
from utils import feature_engineering_kingcounty

def cosine_similarity_scores(X, Y):
    """
    Calculate cosine similarity scores between each row of X and Y.

    Parameters
    ----------
    X : np.ndarray
        The first set of instances.
    Y : np.ndarray
        The second set of instances.

    Returns
    -------
    similarity_scores : np.ndarray
        A similarity score matrix where each entry (i, j) is the cosine similarity
        score between X[i] and Y[j].
    """
    return cosine_similarity(X, Y)

def euclidean_distance_scores(X, Y):
    """
    Calculate Euclidean distance scores between each row of X and Y.

    Parameters
    ----------
    X : np.ndarray
        The first set of instances.
    Y : np.ndarray
        The second set of instances.

    Returns
    -------
    distance_scores : np.ndarray
        A distance score matrix where each entry (i, j) is the Euclidean distance
        between X[i] and Y[j].
    """
    return -np.linalg.norm(X[:, np.newaxis] - Y, axis=2)  # use negative to rank smaller distances higher

def get_similarity_based_candidate_indices(base_model_dir, dataset, sim_feature_names, k, similarity_func):
    """
    Get indices of top k similar candidates for each target property based on a similarity function.

    Parameters
    ----------
    base_model_dir : str
        Path to the previously saved file containing identifiers of 
        target data (test data, e.g., 'test_indices.csv') 
        and candidate data (train and validation data, e.g., 'train_val_indices.csv').
    dataset: str
        Name of the dataset ('kingcounty','hiv','polymer')
    sim_feature_names : list of str
        List of feature names that is used to calculate similarity score.
    k : int
        Number of top similar candidates to retrieve.
    similarity_func : callable
        Function to compute similarity scores between target and candidate features.

    Returns
    -------
    candidate_indices : list of list of int
        Indices in the candidates array for the top k similar candidates of each target explicand.
    """

    # Read the main dataset
    if dataset == "kingcounty":

        df = pd.read_csv("../data/kaggle/kingcountysales/kingco_sales.csv", index_col=0)
        df = feature_engineering_kingcounty(df)   
        # Read the identifiers for the target and candidate datasets
        target_indices = pd.read_csv(os.path.join(base_model_dir, 'test_indices.csv'))
        candidates_indices = pd.read_csv(os.path.join(base_model_dir, 'train_val_indices.csv'))

        # Filter the main dataset to get the target and candidate DataFrames
        df_target = df.merge(target_indices, on=['sale_id', 'pinx'])
        df_candidates = df.merge(candidates_indices, on=['sale_id', 'pinx'])
    else:# other datasets
        df_target = pd.read_csv(os.path.join(base_model_dir, 'X_test.csv'))
        df_candidates = pd.read_csv(os.path.join(base_model_dir, 'X_train_val_combined.csv'))
    
    X_target = df_target[sim_feature_names].to_numpy()
    X_candidates = df_candidates[sim_feature_names].to_numpy()

    # Compute similarity scores
    similarity_scores = similarity_func(X_target, X_candidates)

    # Get the top k candidate indices based on similarity scores
    candidate_indices = [np.argsort(-similarity_scores_row)[:k] for similarity_scores_row in similarity_scores]

    # Debug: Print one example of target and its top k candidates
    if candidate_indices and len(candidate_indices[0]) > 0:
        example_target_idx = 0
        example_target_row = df_target.iloc[example_target_idx]
        target_data = example_target_row.to_dict()
        if dataset == "kingcounty":
            print(f"\nExample Target Property (sale_id: {example_target_row['sale_id']}):")
            print(tabulate(target_data.items(), headers=["Feature", "Value"], tablefmt="pretty"))
            candidate_data_list = []
            print(f"\nTop {k} Similar Candidates for Target Property (sale_id: {example_target_row['sale_id']}):")
            for i, candidate_idx in enumerate(candidate_indices[example_target_idx]):
                candidate_row = df_candidates.iloc[candidate_idx]
                candidate_data = candidate_row[sim_feature_names].to_dict()
                candidate_data["sale_id"] = candidate_row["sale_id"]
                candidate_data_list.append(candidate_data)
            # Convert to tabular format
            formatted_candidate_data = [list(data.values()) for data in candidate_data_list]
            headers = ["sale_id"] + sim_feature_names
            print(tabulate(formatted_candidate_data, headers=headers, tablefmt="pretty"))
        else:
            print("No print example for dataset other than KingCounty (TODO)")

    return candidate_indices

# # Integration with SHAP pipeline for "Pairwise-similarity"
# if explain_method == "Pairwise-similarity":
#     feature_names = X_train_val_combined.columns.to_list()
    
#     # Choose the similarity function
#     similarity_func = cosine_similarity_scores  # or euclidean_distance_scores
    
#     candidate_indices = get_similarity_based_candidate_indices(
#         df_target=X_test_transformed,
#         df_candidates=X_train_val_combined,
#         feature_names=feature_names,
#         k=10,
#         similarity_func=similarity_func
#     )
    
#     explicands = X_test_transformed.to_numpy()
#     candidates = X_train_val_combined.to_numpy()
    
#     pairwise_shap_values = pairwise_shapley_explanation(
#         final_model, explicands, candidates, candidate_indices,
#         feature_names=feature_names)
    
#     shap_values = pairwise_shap_values  # For compatibility
# else:
    # raise ValueError("Unsupported explanation method!")