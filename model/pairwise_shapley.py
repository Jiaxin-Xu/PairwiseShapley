import shap
import numpy as np
import pandas as pd
from tqdm import tqdm

from comparable_candidates import similarity
from similar_candidates import cosine_similarity_scores, euclidean_distance_scores

class CustomExplanation(shap.Explanation):
    """
    CustomExplanation extends shap.Explanation to include additional 
    details for pairwise SHAP explanation.

    Attributes
    ----------
    values : np.ndarray
        The SHAP values for the features.
    base_value : float
        The base value for the prediction, typically the predicted value for the candidate.
        Since there is always only one background data in pairwise SHAP computation.
    data : np.ndarray
        Difference between the feature values of the explicand and the candidate.
    feature_names : list of str
        The names of the features.
    """
    def __init__(self, values, base_value, data, feature_names):
        """
        Initializes the CustomExplanation object.

        Parameters
        ----------
        values : np.ndarray
            The SHAP values for the features.
        base_value : float
            The base value for the prediction, typically the predicted value for the candidate.
        data : np.ndarray
            Difference between the feature values of the explicand and the candidate.
        feature_names : list of str
            The names of the features.
        """
        super().__init__(values=values, base_values=base_value, data=data, feature_names=feature_names)


def pairwise_shapley_explanation(final_model, dataset, explicands, candidates, candidate_indices, feature_names, compute_sim=False):
    """
    Calculate pairwise SHAP values by comparing each explicand to its list of comparable candidates.

    Parameters
    ----------
    final_model : object
        The trained model to be explained.
    dataset : str
        Name of the dataset ('kingcounty','hiv','polymer')
    explicands : np.ndarray or pd.DataFrame
        The explicand instances (instances to be explained).
    candidates : np.ndarray or pd.DataFrame
        The candidate instances to compare against.
    candidate_indices : list of list of int
        Indices in the candidates array for comparable candidates of each explicand.
    feature_names : list of str
        List of feature names.
    compute_sim : bool, optional (default=False)
        Whether to compute similarity scores.

    Returns
    -------
    explanation : shap.Explanation
        SHAP value explanations for each explicand, compared to each candidate.
    similarity_scores : dict
        A dictionary containing cosine, euclidean, and comps similarity scores for each pair of explicand and candidate.
    """
    all_shap_values = []
    all_base_values = []
    all_data_differences = []
    all_feature_names = feature_names
    
    similarity_scores = {
        'cos_sim_scores': [],
        'euc_sim_scores': [],
        'comps_sim_scores': []
    } if compute_sim else None

    # for comps_sim_scores
    feature_dict = {
    # 'land_val': {'diff_type': "relative", 'diff_at_half_dist': 0.1, 'guardrail_bounds': None},
    'sqft' : {'diff_type': "relative", 'diff_at_half_dist': 0.1, 'guardrail_bounds': None},
    'bath_full': {'diff_type': "absolute", 'diff_at_half_dist': 1, 'guardrail_bounds': None},
    'bath_comb': {'diff_type': "absolute", 'diff_at_half_dist': 1, 'guardrail_bounds': None},
    'grade': {'diff_type': "absolute", 'diff_at_half_dist': 2, 'guardrail_bounds': None},
    }
    
    # Process each explicand
    for explicand_idx, explicand in enumerate(tqdm(explicands, desc="Processing explicands")):
        for candidate_idx in candidate_indices[explicand_idx]:
            candidate = candidates[candidate_idx].reshape(1, -1)
            
            explainer = shap.KernelExplainer(final_model.predict,#final_model.predict, 
                                             candidate, feature_names=feature_names)
            shap_values = explainer.shap_values(explicand.reshape(1, -1), silent=True)
            predicted_candidate_value = final_model.predict(candidate).item() # final_model.predict(candidate).item()
            # print("TEST----SHAPE of SHAP and Prediction@@@@@@")
            # print(shap_values.shape)
            # print(predicted_candidate_value)
            
            data_difference = explicand - candidate.flatten()
            
            all_shap_values.append(shap_values.flatten())  # Flatten in case it's a single-dimension array
            all_base_values.append(predicted_candidate_value)
            all_data_differences.append(data_difference)

            if compute_sim:
                # Calculate similarity scores
                cos_sim = cosine_similarity_scores(explicand.reshape(1, -1), candidate).flatten()[0]
                euc_sim = euclidean_distance_scores(explicand.reshape(1, -1), candidate).flatten()[0]
                similarity_scores['cos_sim_scores'].append(cos_sim)
                similarity_scores['euc_sim_scores'].append(euc_sim)
                if dataset == "kingcounty":
                    comps_sim = similarity(
                        pd.DataFrame([candidate.flatten()], columns=feature_names), 
                        pd.Series(explicand, index=feature_names), 
                        feature_dict=feature_dict, 
                        n_candidates=1
                    )[0]
                    similarity_scores['comps_sim_scores'].append(comps_sim)
    
    # Convert lists to numpy arrays for shap.Explanation
    all_shap_values = np.array(all_shap_values)
    all_base_values = np.array(all_base_values)
    all_data_differences = np.array(all_data_differences)

    # Create a unified shap.Explanation object
    explanation = shap.Explanation(
        values=all_shap_values,
        base_values=all_base_values,
        data=all_data_differences,
        feature_names=all_feature_names
    )
    
    return explanation, similarity_scores


# Example Usage
# Assuming `final_model`, `X_test`, and `candidates` are defined, with `candidate_indices` giving the indices of comparable candidates
# for each explicand in `X_test`.

# explicands = X_test.to_numpy()  # Or use X_test.values for pandas DataFrame
# candidates = X_train_val_combined.to_numpy()  # Or similar for your candidate instances

# # Example: candidate_indices where each explicand gets 10 candidates
# candidate_indices = [np.random.choice(range(candidates.shape[0]), 10, replace=False) for _ in range(explicands.shape[0])]

# pairwise_shap_values = pairwise_shapley_explanation(final_model, explicands, candidates, candidate_indices, feature_names=X_train_val_combined.columns.to_list())

# # Print first explicand comparisons
# print(pairwise_shap_values[0])

# # Integration into the existing pipeline
# if explain_method == "Pairwise":
#     explicands = X_test.to_numpy()
#     candidates = X_train_val_combined.to_numpy()
#     candidate_indices = [np.random.choice(range(candidates.shape[0]), 10, replace=False) for _ in range(explicands.shape[0])]
#     pairwise_shap_values = pairwise_shapley_explanation(final_model, explicands, candidates, candidate_indices, feature_names=X_train_val_combined.columns.to_list())
#     shap_values = pairwise_shap_values  # For compatibility
# else:
#     raise ValueError("Unsupported explanation method!")