import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
import os
from scipy.stats import entropy, spearmanr, gaussian_kde
import pickle
import itertools
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from utils import setup_logging
from tabulate import tabulate
import argparse
from comparable_candidates import similarity
from similar_candidates import cosine_similarity_scores, euclidean_distance_scores


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Post-processing on explaination of model for King County home price (and two more molecule/material datasets) prediction.')
    parser.add_argument('--model_version', type=str, default='v0', 
                        choices=['v3', 'v2', 'v1', 'v0'
                                ], 
                        help='base model version to use (King County dataset only); \
                        v0 - original features; \
                        v1 - bath-combine; \
                        v2 - remove some features based on shapley value; \
                        v3 - further remove to 30 features in total')
    parser.add_argument('--dataset', type=str, default='kingcounty', 
                        choices=['kingcounty', \
                                 'hiv', \
                                 'polymer'
                                ], 
                        help='Dataset to use')
    return parser.parse_args()

def load_shap_values(paths):
    shap_values_list = []
    feature_values = None  # Initialize to None
    
    for path in paths:
        with open(path, 'rb') as file:
            shap_data = pickle.load(file)
            shap_values_list.append(shap_data.values)
            if feature_values is None:
                feature_values = shap_data.data # feature values are the same for different standard (non-pairwise) shapley method
    
    return shap_values_list, feature_values

def load_shap_values_pair(paths):
    shap_values_list = []
    feature_values_list = []  # feature values are different for different pairwise method
    
    for path in paths:
        with open(path, 'rb') as file:
            shap_data = pickle.load(file)
            shap_values_list.append(shap_data.values)
            feature_values_list.append(shap_data.data)
    
    return shap_values_list, feature_values_list

def load_sim_scores_pair(paths):
    sim_scores_list = []
    
    for path in paths:
        with open(path, 'rb') as file:
            sim_scores = pickle.load(file)
            sim_scores_list.append(sim_scores)
    
    return sim_scores_list

def plot_shap_distributions(shap_values_list, feature_names, method_names, output_dir):
    num_features = shap_values_list[0].shape[1]
    sns.set(style="whitegrid")

    os.makedirs(output_dir, exist_ok=True)

    # seperate dist'n for each feature
    for feature_idx in range(num_features):
        plt.figure(figsize=(10, 4))
        for i, shap_values in enumerate(shap_values_list):
            sns.kdeplot(shap_values[:, feature_idx], label=method_names[i])

        plt.title(f'Distribution of SHAP values for Feature {feature_names[feature_idx]}')
        plt.xlabel('SHAP value')
        plt.ylabel('Density')
        plt.legend()
        plot_path = os.path.join(output_dir, f'shap_distribution_feature_{feature_names[feature_idx]}.png')
        plt.tight_layout() 
        plt.savefig(plot_path)
        plt.close()
        
    # combined violin plot for all features
    # Prepare data for plotting
    data = []
    for method_idx, shap_values in enumerate(shap_values_list):
        for feature_idx in range(num_features):
            for shap_value in shap_values[:, feature_idx]:
                data.append([feature_names[feature_idx], shap_value, method_names[method_idx]])
    
    # Convert to a DataFrame for seaborn
    df = pd.DataFrame(data, columns=["Feature", "SHAP value", "Method"])
    
    plt.figure(figsize=(30,10))
    sns.violinplot(x="Feature", y="SHAP value", hue="Method", data=df, split=False, inner="quartile", 
                   palette="muted", density_norm="width")

    plt.title('Distribution of SHAP values for all features')
    plt.xlabel('Feature')
    plt.ylabel('SHAP value')
    plt.legend()
    plt.xticks(rotation=90)  # Rotate feature names if too many
    plot_path = os.path.join(output_dir, f'shap_distribution_all.png')
    plt.tight_layout() 
    plt.savefig(plot_path)
    plt.close()
    print("Plot shap distributions!")

def compute_normalized_shap_differences(shap_values_list, feature_values, method_names, feature_names, num_pairs=10, random_seed=42):
    """
    Compute normalized SHAP value differences and also output shap_values_list_diff and feature_values_list_diff.

    Parameters
    ----------
    shap_values_list : list of np.ndarray
        List of SHAP values arrays from different methods.
    feature_values : np.ndarray
        The original feature values.
    method_names : list of str
        The names of the methods used to calculate SHAP values.
    feature_names : list of str
    
    num_pairs : int, optional
        Number of random pairs for normalization calculation.
    random_seed : int, optional
        Seed for the random number generator to ensure consistent results (default is 42).

    Returns
    -------
    normalized_diffs : dict
        A dictionary containing normalized SHAP value differences for each method and each feature.
    shap_values_list_diff : list of np.ndarray
        List of SHAP value difference arrays for different methods.
    feature_values_list_diff : list of np.ndarray
        List of feature difference arrays for different methods.
    similarity_scores : dict
        A dictionary containing cosine, euclidean, and comps [DEPRECATED] similarity scores for each pair of homes.
    """

    np.random.seed(random_seed)
    random.seed(random_seed)
    
    num_homes = feature_values.shape[0]
    num_features = feature_values.shape[1]
    
    normalized_diffs = {method: {feature: [] for feature in range(num_features)} for method in method_names}
    
    shap_values_list_diff = []
    feature_values_list_diff = []

    similarity_scores_list = [] # [method1, method2, ...]

    # # for comps_sim_scores
    # feature_dict = {
    # # 'land_val': {'diff_type': "relative", 'diff_at_half_dist': 0.1, 'guardrail_bounds': None},
    # 'sqft' : {'diff_type': "relative", 'diff_at_half_dist': 0.1, 'guardrail_bounds': None},
    # 'bath_full': {'diff_type': "absolute", 'diff_at_half_dist': 1, 'guardrail_bounds': None},
    # 'bath_comb': {'diff_type': "absolute", 'diff_at_half_dist': 1, 'guardrail_bounds': None},
    # 'grade': {'diff_type': "absolute", 'diff_at_half_dist': 2, 'guardrail_bounds': None},
    # }

    for idx, shap_values in enumerate(shap_values_list):
        shap_diffs = []
        feature_diffs = []
        
        similarity_scores = {
        'cos_sim_scores': [],
        'euc_sim_scores': [],
        'comps_sim_scores': []
    }   
        for home_idx in range(num_homes):
            for _ in range(num_pairs):
                random_idx = random.randint(0, num_homes - 1)
                feature_diff = feature_values[home_idx, :] - feature_values[random_idx, :]
                shap_diff = shap_values[home_idx, :] - shap_values[random_idx, :]
                
                # Store the diffs
                shap_diffs.append(shap_diff)
                feature_diffs.append(feature_diff)
                
                # Calculate normalized differences
                for feature_idx in range(num_features):
                    if feature_diff[feature_idx] != 0:
                        normalized_diff = shap_diff[feature_idx] / feature_diff[feature_idx]
                        normalized_diffs[method_names[idx]][feature_idx].append(normalized_diff)

                # Calculate similarity scores
                similarity_scores['cos_sim_scores'].append(cosine_similarity_scores(feature_values[home_idx, :].reshape(1, -1), feature_values[random_idx, :].reshape(1, -1)).flatten()[0])
                similarity_scores['euc_sim_scores'].append(euclidean_distance_scores(feature_values[home_idx, :].reshape(1, -1), feature_values[random_idx, :].reshape(1, -1)).flatten()[0])
                # comps_scores.append(similarity(
                #     pd.DataFrame(feature_values,columns=feature_names),
                #     pd.Series(feature_values[home_idx, :],index=feature_names), 
                #     feature_dict=feature_dict, 
                #     n_candidates=1)[0])
        
        # Convert lists to numpy arrays and store in the lists
        shap_values_list_diff.append(np.array(shap_diffs))
        feature_values_list_diff.append(np.array(feature_diffs))
        similarity_scores_list.append(similarity_scores)
        
        # similarity_scores['cos_sim_scores'].append(np.array(cos_scores))
        # similarity_scores['euc_sim_scores'].append(np.array(euc_scores))
        # similarity_scores['comps_sim_scores'].append(np.array(comps_scores))
    
    return normalized_diffs, shap_values_list_diff, feature_values_list_diff, similarity_scores_list

def compute_normalized_shap_differences_pairwise(shap_values_list, feature_values_list, method_names):
    """
    Compute normalized SHAP value differences for pairwise SHAP explanations.

    Parameters
    ----------
    shap_values_list : list of np.ndarray
        List of SHAP values arrays from different methods. Each array is of shape (num_homes, num_features).
    feature_values_list : list of np.ndarray
        List of feature difference arrays from different methods. Each array is of shape (num_homes, num_features).
    method_names : list of str
        The names of the methods used to calculate SHAP values.

    Returns
    -------
    normalized_diffs : dict
        A dictionary containing normalized SHAP value differences for each method and each feature.
    """
    num_methods = len(shap_values_list)
    num_features = shap_values_list[0].shape[1]

    # Initialize dictionary to store normalized differences
    normalized_diffs = {method: {feature: [] for feature in range(num_features)} for method in method_names}

    # Loop through each method
    for method_idx in range(num_methods):
        method = method_names[method_idx]
        shap_values = shap_values_list[method_idx]
        feature_values = feature_values_list[method_idx]

        # Loop through each home (explicand)
        for home_idx in range(shap_values.shape[0]):
            
            # Loop through each feature
            for feature_idx in range(num_features):
                feature_diff = feature_values[home_idx, feature_idx]
                shap_diff = shap_values[home_idx, feature_idx]
                
                # Calculate and store normalized difference
                if feature_diff != 0:
                    normalized_diff = shap_diff / feature_diff
                    normalized_diffs[method][feature_idx].append(normalized_diff)
    
    return normalized_diffs


def plot_normalized_distribution(shap_values_list, feature_values,
                                 shap_values_list_pair, feature_values_pair,
                                 feature_names, method_names, method_names_pair, output_dir):
    
    normalized_diffs_pair = compute_normalized_shap_differences_pairwise(shap_values_list_pair, feature_values_pair, method_names_pair)
    # for non-pairwise method, create random pairs
    num_pairs = 50
    normalized_diffs, shap_values_list_diff, feature_values_list_diff, similarity_scores_list = compute_normalized_shap_differences(shap_values_list, 
                                                                                                                               feature_values, method_names, 
                                                                                                                               feature_names, num_pairs=num_pairs)
    # save nonpairwise sim scores for nonpairwise method.
    try:
        with open(os.path.join(output_dir, 'nonpairwise_similarity_scores.pkl'), 'wb') as file:
            pickle.dump(similarity_scores, file)
        print(f"Similarity scores have been saved to {output_dir}.")
    except NameError:
        pass
        
    # combine two dicts into one
    normalized_diffs.update(normalized_diffs_pair)
    
    # print(normalized_diffs)
    num_features = len(feature_names)
    sns.set(style="whitegrid")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    data = []
    for method_name in normalized_diffs.keys():
        for feature_idx in normalized_diffs[method_name].keys():
            for norm_diff in normalized_diffs[method_name][feature_idx]:
                data.append([feature_names[feature_idx], norm_diff, method_name])
    
    # Convert to a DataFrame for seaborn
    df = pd.DataFrame(data, columns=["Feature", "Normalized SHAP value", "Method"])
    # print(df)

    # combined plot
    plt.figure(figsize=(30,10))
    sns.violinplot(x="Feature", y="Normalized SHAP value", hue="Method", data=df, split=False, inner="quartile", 
                   palette="muted", density_norm="width")

    plt.title('Normalized Distribution of SHAP values for all features')
    plt.xlabel('Feature')
    plt.ylabel('Normalized SHAP value')
    plt.legend()
    plt.xticks(rotation=90)  # Rotate feature names if too many
    plot_path = os.path.join(output_dir, 'normalized_shap_distribution_all.png')
    plt.tight_layout() 
    plt.savefig(plot_path)
    plt.close()

    # seperate plot on individual features
    for feature_idx in range(num_features):
        plt.figure(figsize=(10, 4))
        for method_name in normalized_diffs.keys():
            sns.kdeplot(normalized_diffs[method_name][feature_idx], label=method_name)
            print(f"{method_name}-{feature_names[feature_idx]}: mean = {np.mean(normalized_diffs[method_name][feature_idx])}, std={np.std(normalized_diffs[method_name][feature_idx])}")
            # sns.kdeplot(shap_values[:, feature_idx], label=method_names[i])

        plt.title(f'Normalized distribution of SHAP values for Feature {feature_names[feature_idx]}')
        plt.xlabel('Normalized SHAP value')
        plt.ylabel('Density')
        plt.legend()
        plot_path = os.path.join(output_dir, f'normalized_shap_distribution_{feature_names[feature_idx]}.png')
        plt.tight_layout() 
        plt.savefig(plot_path)
        plt.close()

            # Check if the feature name contains "sqft" and create an additional plot with adjusted x-axis
        methods_to_plot = ['Marginal-all', 'Baseline-median', 'Uniform']
        pairwise_methods = ['Pairwise-random', 'Pairwise-comps', 'Pairwise-sim']  # Replace these with the actual pairwise method names if different
        methods_to_plot.extend(pairwise_methods)
        if "sqft" in feature_names[feature_idx]:
            plt.figure(figsize=(10, 4))
            for method_name in methods_to_plot:
                # sns.kdeplot(normalized_diffs[method_name][feature_idx], label=method_name)
                # Clip the SHAP values to the range -700 to 700
                clipped_values = np.clip(normalized_diffs[method_name][feature_idx], -700, 700)
                if method_name in pairwise_methods:
                    sns.kdeplot(clipped_values, label=method_name, linestyle='-')
                else:
                    sns.kdeplot(clipped_values, label=method_name, linestyle='--')
                # sns.kdeplot(clipped_values, label=method_name)
                print(f"CLIPPED-{method_name}-{feature_names[feature_idx]}: mean = {np.mean(clipped_values)}, std={np.std(clipped_values)}")
                
            plt.xlim(-500, 500)
            plt.title(f'Adjusted scale normalized distribution of SHAP values for Feature {feature_names[feature_idx]}')
            plt.xlabel('Normalized SHAP value')
            plt.ylabel('Density')
            plt.legend()
            plot_path = os.path.join(output_dir, f'normalized_shap_distribution_{feature_names[feature_idx]}_adjusted_submethods.png')
            plt.tight_layout() 
            plt.savefig(plot_path)
            plt.close()
    print("Plot normalized distribution!")
    return normalized_diffs, shap_values_list_diff, feature_values_list_diff, similarity_scores_list

# Function to compute smoothed KL divergence
def kl_divergence_smoothed(p_pdf, q_pdf, epsilon=1e-10):
    p_pdf = np.clip(p_pdf, epsilon, None)  # Ensure p_pdf is at least epsilon
    q_pdf = np.clip(q_pdf, epsilon, None)  # Ensure q_pdf is at least epsilon
    kl_div = entropy(p_pdf, q_pdf)
    return kl_div

def js_divergence_smoothed(p_pdf, q_pdf, epsilon=1e-10):
    """
    The Jensen-Shannon divergence (JSD) is a symmetrical measure based on KL-divergence
    """
    p_pdf = np.clip(p_pdf, epsilon, None)  # Ensure p_pdf is at least epsilon
    q_pdf = np.clip(q_pdf, epsilon, None)  # Ensure q_pdf is at least epsilon
    # Calculate the average distribution M
    m_pdf = 0.5 * (p_pdf + q_pdf)
    
    # Calculate KL divergence components
    kl_div_p_m = entropy(p_pdf, m_pdf)
    kl_div_q_m = entropy(q_pdf, m_pdf)
    
    # Calculate JSD
    js_div = 0.5 * (kl_div_p_m + kl_div_q_m)
    return js_div

def compute_kl_divergence_and_correlation(shap_values_list, normalized_shap_diffs, feature_values, method_names, symmetry=True):
    kl_divergences = {'original': {}, 'normalized': {}}
    correlations = {}
    
    num_features = feature_values.shape[1]

    # Rank Correlation on Original Feature Values
    cor_matrix = np.ones((num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            if i != j:
                corr, _ = spearmanr(feature_values[:, i], feature_values[:, j])
                cor_matrix[i, j] = corr
    correlations = cor_matrix

    # for idx, shap_values in enumerate(shap_values_list):
    #     method = method_names[idx]

    #     # KL-divergence: 
    #     # a type of statistical "distance": a measure of how one probability distribution P is different from a second, reference probability distribution Q
    #     # a non-negative real number, with value 0 if and only if the two distributions in question are identical
    #     kl_matrix_original = np.zeros((num_features, num_features))
    #     kl_matrix_normalized = np.zeros((num_features, num_features))
        
    #     for i in range(num_features):
    #         for j in range(num_features):
    #             if i != j:
    #                 # Compute KL divergence for original SHAP values
    #                 p_samples = shap_values[:, i]
    #                 q_samples = shap_values[:, j]

    #                 # estimate the probability density function (PDF) of a random variable in a non-parametric way using Gaussian kernels;
    #                 # The estimation works best for a unimodal distribution; bimodal or multi-modal distributions tend to be oversmoothed.
    #                 kde_p = gaussian_kde(p_samples) 
    #                 kde_q = gaussian_kde(q_samples)
                    
    #                 x = np.linspace(min(p_samples.min(), q_samples.min()), max(p_samples.max(), q_samples.max()), 1000)
    #                 p_pdf = kde_p(x)
    #                 q_pdf = kde_q(x)
                    
    #                 p_pdf /= p_pdf.sum()
    #                 q_pdf /= q_pdf.sum()
                    
    #                 # kl_div = entropy(p_pdf, q_pdf)
    #                 if symmetry:
    #                     kl_div = js_divergence_smoothed(p_pdf, q_pdf, epsilon=1e-10) # use js-divergence
    #                 else:
    #                     kl_div = kl_divergence_smoothed(p_pdf, q_pdf, epsilon=1e-10)
    #                 kl_matrix_original[i, j] = kl_div

    #                 # Compute KL divergence for normalized SHAP values
    #                 norm_p_samples = np.array(normalized_shap_diffs[method][i])
    #                 norm_q_samples = np.array(normalized_shap_diffs[method][j])
                    
    #                 kde_norm_p = gaussian_kde(norm_p_samples)
    #                 kde_norm_q = gaussian_kde(norm_q_samples)
                    
    #                 norm_x = np.linspace(min(norm_p_samples.min(), norm_q_samples.min()), max(norm_p_samples.max(), norm_q_samples.max()), 1000)
    #                 norm_p_pdf = kde_norm_p(norm_x)
    #                 norm_q_pdf = kde_norm_q(norm_x)
                    
    #                 norm_p_pdf /= norm_p_pdf.sum()
    #                 norm_q_pdf /= norm_q_pdf.sum()
                    
    #                 # kl_div_norm = entropy(norm_p_pdf, norm_q_pdf)
    #                 if symmetry:
    #                     kl_div_norm = js_divergence_smoothed(norm_p_pdf, norm_q_pdf, epsilon=1e-10)# use js-divergence
    #                 else:
    #                     kl_div_norm = kl_divergence_smoothed(norm_p_pdf, norm_q_pdf, epsilon=1e-10)
    #                 kl_matrix_normalized[i, j] = kl_div_norm
        
    #     kl_divergences['original'][method] = kl_matrix_original
    #     kl_divergences['normalized'][method] = kl_matrix_normalized

    # print("Compute KL-D or JS-D!")
    
    return kl_divergences, correlations


def plot_custom_heatmap_corr(matrix, feature_names, title, output_path):
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    figsize = (25, 25)

    sns.set(style="white")
    plt.figure(figsize=figsize)
    
    # Custom heatmap with mask
    ax = sns.heatmap(matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", 
                     cbar_kws={"shrink": 0.5, "aspect": 10}, annot_kws={"size": 14},
                     xticklabels=feature_names, yticklabels=feature_names)
    
    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap("coolwarm")
    
    # Use circles on the upper triangle to denote absolute value size and color for pos/neg
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if mask[i, j]:
                abs_val = np.abs(matrix[i, j])
                color = cmap(norm(matrix[i, j]))
                
                # Scale font size by absolute value
                font_size = abs_val / np.max(np.abs(matrix)) * 30
                
                plt.text(j + 0.5, i + 0.5, '•', ha='center', va='center', color=color, fontsize=font_size)
    
    plt.title(title, fontsize=32)
    plt.xlabel('Features', fontsize=24)
    plt.ylabel('Features', fontsize=24)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(rotation=0, fontsize=20)

    # Adjust color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_kl_and_correlation_heatmaps(kl_divergences, correlations, feature_names, method_names, output_dir):
    num_features = len(feature_names)
    sns.set(style="whitegrid")
    
    os.makedirs(output_dir, exist_ok=True)
    
    ## Plot correlation heatmap
    
    # plt.figure(figsize=(30, 30))
    # sns.heatmap(correlations, xticklabels=feature_names, yticklabels=feature_names, cmap="coolwarm", annot=True)
    # plt.title('Rank Correlation Heatmap')
    # plt.xlabel('Features')
    # plt.ylabel('Features')
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=0)
    # plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
    # plt.tight_layout() 
    # plt.savefig(plot_path)
    # plt.close()
    
    cor_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plot_custom_heatmap_corr(correlations, feature_names, 'Rank Correlation Heatmap', cor_path)
    
    for method in method_names:
        kl_matrix_original = kl_divergences['original'][method]
        kl_matrix_normalized = kl_divergences['normalized'][method]
        
        plt.figure(figsize=(25, 25))
        sns.heatmap(kl_matrix_original, annot=True, cmap="viridis", fmt=".2f", 
                     cbar_kws={"shrink": 0.5, "aspect": 10}, annot_kws={"size": 14},
                     xticklabels=feature_names, yticklabels=feature_names)
        plt.title(f'KL/JS Divergence Heatmap (Original SHAP Values) for {method}')
        plt.xlabel('Features', fontsize=24)
        plt.ylabel('Features', fontsize=24)
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(rotation=0, fontsize=20)
        plot_path = os.path.join(output_dir, f'kl_divergence_heatmap_original_{method}.png')
        plt.tight_layout() 
        plt.savefig(plot_path)
        plt.close()
        
        plt.figure(figsize=(25, 25))
        sns.heatmap(kl_matrix_normalized, annot=True, cmap="viridis", fmt=".2f", 
                     cbar_kws={"shrink": 0.5, "aspect": 10}, annot_kws={"size": 14},
                     xticklabels=feature_names, yticklabels=feature_names)
        plt.title(f'KL/JS Divergence Heatmap (Normalized SHAP Values) for {method}')
        plt.xlabel('Features', fontsize=24)
        plt.ylabel('Features', fontsize=24)
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(rotation=0, fontsize=20)
        plot_path = os.path.join(output_dir, f'kl_divergence_heatmap_normalized_{method}.png')
        plt.tight_layout() 
        plt.savefig(plot_path)
        plt.close()
    print("Plot kl/js -d and correlation heatmaps!")



def find_highly_correlated_feature_groups(correlations, feature_names, threshold=0.3):
    num_features = correlations.shape[0]
    feature_indices = range(num_features)
    max_subset_size = num_features

    # Initialize a list to store the largest highly correlated groups
    high_corr_groups = []

    # Check all subset sizes from 3 to the maximum possible size
    for subset_size in range(3, max_subset_size + 1):
        for subset in itertools.combinations(feature_indices, subset_size):
            subset_corr = correlations[np.ix_(subset, subset)]
            # Compare absolute correlation values with the threshold
            if np.all(np.abs(subset_corr[np.triu_indices_from(subset_corr, k=1)]) > threshold):
                subset_features = [feature_names[i] for i in subset]
                
                # Check if this subset is not a subset of any existing group
                is_subset = any(set(subset_features).issubset(set(existing_group)) for existing_group in high_corr_groups)
                
                if not is_subset:
                    high_corr_groups.append(subset_features)

    # Filter out smaller groups that are subset of a larger group one more time
    final_high_corr_groups = []
    for group in high_corr_groups:
        if not any(set(group).issubset(set(existing_group)) and group != existing_group for existing_group in high_corr_groups):
            final_high_corr_groups.append(group)

    return final_high_corr_groups

def calculate_shap_value_proportions_ingroup(shap_values_list, feature_names, high_corr_groups, method_names):
    
    shap_value_proportions = {method: {} for method in method_names}
    
    for method_idx, shap_values in enumerate(shap_values_list):
        method = method_names[method_idx]
        for group in high_corr_groups:
            group_indices = [feature_names.index(f) for f in group]
            group_shap_values = shap_values[:, group_indices]

            # # Find the most negative value in each sample's group
            # most_negative_value = np.min(group_shap_values, axis=1, keepdims=True)
            
            # # Shift all values by adding the absolute value of the most negative value
            # shifted_shap_values = group_shap_values + np.abs(most_negative_value)

            # Compute the absolute sum of SHAP values
            group_total_shap_values = np.abs(group_shap_values).sum(axis=1, keepdims=True)
            # Avoid division by zero by setting sums that are zero to a small value (epsilon)
            group_total_shap_values[group_total_shap_values == 0] = 1e-10
            # Calculate proportions using absolute values
            proportions = np.abs(group_shap_values) / group_total_shap_values
            
            
            # group_total_shap_values = group_shap_values.sum(axis=1, keepdims=True)
            # proportions = group_shap_values / group_total_shap_values  # Proportion for each feature in the group

            shap_value_proportions[method][tuple(group)] = proportions
    
    return shap_value_proportions

def calculate_norm_shap_value_proportions_ingroup(norm_shap_values, feature_names, high_corr_groups, method_names):
    
    shap_value_proportions = {method: {} for method in method_names}
    
    # for method_idx, shap_values in enumerate(shap_values_list):
    for method_name in norm_shap_values.keys():
        # method = method_names[method_idx]
        for group in high_corr_groups:
            group_indices = [feature_names.index(f) for f in group]
            group_shap_values = [norm_shap_values[method_name][idx] for idx in group_indices]
            group_shap_values = np.array(group_shap_values)
            # group_shap_values = shap_values[:, group_indices]

            # Compute the absolute sum of SHAP values
            group_total_shap_values = np.abs(group_shap_values).sum(axis=0, keepdims=True)
            # Avoid division by zero by setting sums that are zero to a small value (epsilon)
            group_total_shap_values[group_total_shap_values == 0] = 1e-10
            # Calculate proportions using absolute values
            proportions = np.abs(group_shap_values) / group_total_shap_values
            proportions = proportions.T
            
            # group_total_shap_values = group_shap_values.sum(axis=1, keepdims=True)
            # proportions = group_shap_values / group_total_shap_values  # Proportion for each feature in the group

            shap_value_proportions[method][tuple(group)] = proportions
    
    return shap_value_proportions
    
# def plot_shap_value_proportions(shap_value_proportions, output_dir):
#     sns.set(style="whitegrid")
    
#     for method, groups in shap_value_proportions.items():
#         for group, proportions in groups.items():
#             plt.figure(figsize=(10, 6))
#             group = list(group)
#             proportions_df = pd.DataFrame(proportions, columns=group)
#             sns.violinplot(data=proportions_df)
            
#             plt.title(f'Proportion of SHAP Values for {method}' + "\n" + " and Group: " + ", ".join(group))
#             plt.xlabel('Feature')
#             plt.ylabel('Proportion of SHAP Value')
#             plt.xticks(rotation=45)
#             plt.tight_layout()
#             plot_path = os.path.join(output_dir, f'violin_shap_proportion_{method}_{"_".join(group)}.png')
#             plt.savefig(plot_path)
#             plt.close()

def plot_shap_value_proportions_all_methods(shap_value_proportions, feature_names, method_names, output_dir, norm=False):
    sns.set(style="whitegrid")

    combined_data = []
    for group in shap_value_proportions[method_names[0]].keys():  # Assuming all methods have the same groups
        for method in method_names:
            group_proportions = shap_value_proportions[method][tuple(group)]
            for i, feature in enumerate(group):
                for datapoint in group_proportions[:, i]:
                    combined_data.append([feature, datapoint, method, f"Group: {', '.join(group)}"])
    
    # Convert to a DataFrame for seaborn
    df = pd.DataFrame(combined_data, columns=["Feature", "Proportion", "Method", "Group"])

    # Plot each group separately
    groups = df['Group'].unique()
    for i,group in enumerate(groups):
        plt.figure(figsize=(10, 7))

        group_df = df[df['Group'] == group]
        
        # Violin plot with feature grouping and method
        sns.violinplot(x="Feature", y="Proportion", hue="Method", data=group_df, split=False, 
                       inner="quartile", palette="muted", saturation=0.75)
        
        # Annotate the plot with group information (not needed since we're plotting separately)
        plt.legend(title="Method", fontsize=12, title_fontsize=14)
        
        # Add titles and labels
        if norm:
            plt.title(f'Proportion of Normed SHAP Values for {group}', fontsize=16)
        else:
            plt.title(f'Proportion of SHAP Values for {group}', fontsize=16)
        plt.xlabel('Feature', fontsize=14)
        plt.ylabel('Proportion of SHAP Value', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if norm:
            plot_path = os.path.join(output_dir, f'normalized_violin_shap_proportion_G{i}.png')
        else:
            plot_path = os.path.join(output_dir, f'violin_shap_proportion_G{i}.png')
        plt.savefig(plot_path)
        plt.close()
    # plt.figure(figsize=(10, 7))
    
    # # Violin plot with feature grouping and method
    # sns.violinplot(x="Feature", y="Proportion", hue="Method", data=df, split=False, 
    #                inner="quartile", palette="muted", saturation=0.75)
    
    # # Annotate the plot with group information
    # # Add unique labels for groups
    # handles, labels = plt.gca().get_legend_handles_labels()
    # new_labels = []
    # for i, label in enumerate(labels):
    #     if label in new_labels:
    #         new_labels.append(f"{label} - {list(df['Group'].unique())[i % len(df['Group'].unique())]}")
    #     else:
    #         new_labels.append(f"{label}")

    # plt.legend(handles, new_labels, title="Method and Group", fontsize=12, title_fontsize=14)

    # # Add titles and labels
    # if norm:
    #     plt.title(f'Proportion of Normed SHAP Values for All Groups', fontsize=16)
    # else:
    #     plt.title(f'Proportion of SHAP Values for All Groups', fontsize=16)
    # plt.xlabel('Feature', fontsize=14)
    # plt.ylabel('Proportion of SHAP Value', fontsize=14)
    # plt.xticks(rotation=45, fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.tight_layout()
    # if norm:
    #     plot_path = os.path.join(output_dir, 'normalized_violin_shap_proportion_all_methods_all_groups.png')
    # else:
    #     plot_path = os.path.join(output_dir, 'violin_shap_proportion_all_methods_all_groups.png')
    # plt.savefig(plot_path)
    # plt.close()



def analyze_shap_distributions_wcorr(correlations, shap_values_list, normalized_shap_values_list, feature_names, method_names, threshold, output_dir):
    # Step 1: Find highly correlated feature groups
    high_corr_groups = find_highly_correlated_feature_groups(correlations, feature_names, threshold)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"Highly correlated feature groups: {high_corr_groups}")

    # Step 2: Calculate SHAP value proportions
    shap_value_proportions = calculate_shap_value_proportions_ingroup(shap_values_list, feature_names, high_corr_groups, method_names)
    
    # Step 2a: Plot SHAP value proportions for all methods and groups
    plot_shap_value_proportions_all_methods(shap_value_proportions, feature_names, method_names, output_dir)

    # Step 3: Calculate normalized SHAP value proportions
    # normalized_shap_value_proportions = calculate_norm_shap_value_proportions_ingroup(normalized_shap_values_list, feature_names, high_corr_groups, method_names)
    
    # Step 3a: Plot normalized SHAP value proportions for all methods and groups
    # plot_shap_value_proportions_all_methods(normalized_shap_value_proportions, feature_names, method_names, output_dir, norm=True)

def compute_status_counts(shap_diffs, feature_diffs):
    """
    Compute the counts and ratios for all combinations of shap_diffs and feature_diffs statuses:
    (+,+), (+,-), (+,0), (-,+), (-,-), (-,0), (0,+), (0,-), (0,0).

    Parameters
    ----------
    shap_diffs : np.ndarray
        Array of SHAP value differences.
    feature_diffs : np.ndarray
        Array of feature value differences.

    Returns
    -------
    status_counts : dict
        A dictionary containing counts and ratios for all combinations of statuses.
    """
    total_count = len(shap_diffs)
    status_counts = {
        '(+,+)': 0,
        '(+,-)': 0,
        '(+,0)': 0,
        '(-,+)': 0,
        '(-,-)': 0,
        '(-,0)': 0,
        '(0,+)': 0,
        '(0,-)': 0,
        '(0,0)': 0
    }
    
    for shap_diff, feature_diff in zip(shap_diffs, feature_diffs):
        if shap_diff > 0 and feature_diff > 0:
            status_counts['(+,+)'] += 1
        elif shap_diff > 0 and feature_diff < 0:
            status_counts['(+,-)'] += 1
        elif shap_diff > 0 and feature_diff == 0:
            status_counts['(+,0)'] += 1
        elif shap_diff < 0 and feature_diff > 0:
            status_counts['(-,+)'] += 1
        elif shap_diff < 0 and feature_diff < 0:
            status_counts['(-,-)'] += 1
        elif shap_diff < 0 and feature_diff == 0:
            status_counts['(-,0)'] += 1
        elif shap_diff == 0 and feature_diff > 0:
            status_counts['(0,+)'] += 1
        elif shap_diff == 0 and feature_diff < 0:
            status_counts['(0,-)'] += 1
        elif shap_diff == 0 and feature_diff == 0:
            status_counts['(0,0)'] += 1

    # Calculate ratios
    status_ratios = {key: value / total_count for key, value in status_counts.items()}
    
    return status_counts, status_ratios

def plot_monotonicity(df, output_dir):
    
    ## plot 1: monotonicity on all features (bar)
    plt.figure(figsize=(30, 12))
    sns.set(style="whitegrid")
    # Create bar plot with feature on x-axis, spearman_corr on y-axis, and hue on method
    sns.barplot(x='feature', y='spearman_corr', hue='method', data=df)
    plt.title('Spearman Correlation by Feature and Method')
    plt.xticks(rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('Spearman Correlation')
    # plt.ylim((0.5, 1))
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monotonicity_all_bar.png'))

    ## plot 2: monotonicity on important features (bar)
    subset_features = ['sqft', 'grade', 'location_value', 'days_since_sale'] 
    df_imp = df[df['feature'].isin(subset_features)]
    # Step 2: Plot the Spearman correlation for each feature on the same plot
    plt.figure(figsize=(12, 16))
    sns.set(style="whitegrid")    
    # Create bar plot with feature on x-axis, spearman_corr on y-axis, and hue on method
    sns.barplot(x='feature', y='spearman_corr', hue='method', data=df_imp)
    plt.title('Spearman Correlation by Feature and Method')
    plt.xticks(rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('Spearman Correlation')
    # plt.ylim((0.5, 1))
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monotonicity_subset_bar.png'))

    ## plot 3: heatmap, confidently positive features
    subset_features = ['sqft', 'sqft_1', 'sqft_lot', 'grade', 'condition','greenbelt', 
                       'view_rainier', 'view_olympics', 'view_cascades', 'view_territorial',
                        'view_skyline', 'view_sound', 'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other',
                      'wfnt', 'location_value']
    df_pos = df[df['feature'].isin(subset_features)]
    # Pivot the DataFrame to get features on rows and methods on columns
    heatmap_data = df_pos.pivot(index='feature', columns='method', values='spearman_corr')
    # Calculate mean values for each method
    mean_values = heatmap_data.mean().to_frame(name='mean').T
    # Append the mean values as a new row to the heatmap data
    heatmap_data = pd.concat([heatmap_data, mean_values])
    # Plot the heatmap
    plt.figure(figsize=(15, 12))
    sns.set(style="whitegrid")
    # Create the heatmap with annotations
    try:
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="coolwarm", linewidths=.5)
        # Customize the plot
        plt.title('Spearman Correlation Heatmap by Feature and Method')
        plt.xlabel('Method')
        plt.ylabel('Feature')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Add the 'mean' label to the index
        heatmap_data.index = heatmap_data.index[:-1].tolist() + ['mean']
        # Adjust the tick labels
        ax.set_yticklabels(heatmap_data.index)
        plt.savefig(os.path.join(output_dir, 'monotonicity_positive_heatmap.png'))
    except:
        print("Some error creating positive-monotonicity heatmap, probably undefined positive feature list")

    ## plot 4: dummy player heatmap
    df['dummy_player_ratio'] = df['(0,0)_count'] / (df['(+,0)_count'] + df['(-,0)_count'] + df['(0,0)_count'])
    heatmap_data = df.pivot(index='feature', columns='method', values='dummy_player_ratio')
    
    # Define the subset of features you want to include (optional)
    # subset_features = ['feature1', 'feature2', 'feature3', ...]  # Replace with your actual feature names
    # df = df[df['feature'].isin(subset_features)]
    plt.figure(figsize=(15, 12))
    sns.set(style="whitegrid")
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="coolwarm", linewidths=.5)
    plt.title('dummy_player_ratio Heatmap by Feature and Method')
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dummy_player_heatmap.png'))



def plot_conditional_accuracy_heatmap(conditional_spearman_df, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Function to clean category strings and convert to float
    def clean_category_string(category_str):
        category_str = category_str.replace('(', '').replace(')', '').replace(' ', '').replace('[', '').replace(']', '')
        return list(map(float, category_str.split(',')))
    
    # Get unique features
    unique_features = conditional_spearman_df['feature'].unique()
    
    for feature in unique_features:
        # Filter DataFrame for the current feature
        feature_df = conditional_spearman_df[conditional_spearman_df['feature'] == feature]

        # Clean and convert category strings
        categories_cleaned = feature_df['category'].apply(clean_category_string)
        feature_df[['category_start', 'category_end']] = pd.DataFrame(categories_cleaned.tolist(), index=feature_df.index)

        # Create a combined index for 'feature' and sorted 'category'
        feature_df['feature_category'] = feature_df['feature'] + ":" + feature_df['category'].astype(str)

        # Pivot the dataframe for heatmap
        heatmap_data = feature_df.pivot(index='feature_category', columns='method', values='spearman_corr')

        # Re-sort the pivot table by extracting the numeric bin edges
        def sort_index(feature_category):
            feature_name, category_str = feature_category.split(':')
            start, end = clean_category_string(category_str)
            return start

        sorted_index = sorted(heatmap_data.index, key=sort_index)
        heatmap_data = heatmap_data.reindex(sorted_index)    

        plt.figure(figsize=(10, 8))
        sns.set(style="whitegrid")
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="coolwarm", linewidths=.5)
        
        plt.title(f'Conditional Spearman Correlation by Method and Category/Bin for {feature}')
        plt.xlabel('Method')
        plt.ylabel('Feature Diff - Bins')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'conditional_spearman_heatmap_{feature}.png'))
        plt.close()  # Close the figure to avoid memory issues

def plot_conditional_accuracy_heatmap_sim(conditional_spearman_df_sim, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Function to clean category strings and convert to float
    def clean_category_string(category_str):
        category_str = category_str.replace('(', '').replace(')', '').replace(' ', '').replace('[', '').replace(']', '')
        return list(map(float, category_str.split(',')))
    
    # Get unique features
    unique_features = conditional_spearman_df_sim['feature'].unique()
    
    for feature in unique_features:
        # Filter DataFrame for the current feature
        feature_df = conditional_spearman_df_sim[conditional_spearman_df_sim['feature'] == feature]

        # Clean and convert category strings
        categories_cleaned = feature_df['category'].apply(clean_category_string)
        feature_df[['category_start', 'category_end']] = pd.DataFrame(categories_cleaned.tolist(), index=feature_df.index)

        # Create a combined index for 'feature' and sorted 'category'
        feature_df['sim_category'] = "sim_score" + ":" + feature_df['category'].astype(str)

        # Pivot the dataframe for heatmap
        heatmap_data = feature_df.pivot(index='sim_category', columns='method', values='spearman_corr')

        # Re-sort the pivot table by extracting the numeric bin edges
        def sort_index(feature_category):
            feature_name, category_str = feature_category.split(':')
            start, end = clean_category_string(category_str)
            return start

        sorted_index = sorted(heatmap_data.index, key=sort_index)
        heatmap_data = heatmap_data.reindex(sorted_index)    

        plt.figure(figsize=(10, 8))
        sns.set(style="whitegrid")
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="Blues", linewidths=.5)
        
        plt.title(f'Conditional Spearman Correlation by Method and Category/Bin for {feature}')
        plt.xlabel('Method')
        plt.ylabel('Similarity Score - Bins')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'conditional_spearman_heatmap_{feature}_sim.png'))
        plt.close()  # Close the figure to avoid memory issues

def get_unified_bins(feature_name, feature_values_list, bins=5):

    PREDEFINED_BINS = {
        'sqft': [-np.inf, -1000, -500, 0, 500, 1000, np.inf],
        'sqft_1': [-np.inf, -1000, -500, 0, 500, 1000, np.inf],
        'sqft_fbsmt': [-np.inf, -1000, -500, 0, 500, 1000, np.inf],
        'sqft_lot': [-np.inf, -600000, -300000, 0, 300000, 600000, np.inf],
        'grade': [-np.inf, -5, -2, 0, 2, 5, np.inf],
        'bath_comb': [-np.inf, -3, -1, 0, 1, 3, np.inf],
        'beds': [-np.inf, -3, -1, 0, 1, 3, np.inf],
        'condition': [-np.inf, -3, -1, 0, 1, 3, np.inf],
        'fbsmt_grade': [-np.inf, -3, -1, 0, 1, 3, np.inf],
        'home_age': [-np.inf, -100, -50, -25, 0, 25, 50, np.inf],
        'reno_age': [-np.inf, -100, -50, -25, 0, 25, 50, np.inf],
        'location_value': [-np.inf, -800, -500, -200, 0, 200, 500, 800, np.inf],
        'noise_traffic': [-np.inf, -2, 0, 2, np.inf],
        'stories': [-np.inf, -2, 0, 2, np.inf],
        'wfnt': [-np.inf, -8, -5, -2, 0, 2, 5, 8, np.inf],
    }
    
    # Add view_xxx features to predefined bins
    for view_feature in ['view_rainier', 'view_olympics', 'view_cascades', 'view_territorial',
                         'view_skyline', 'view_sound', 'view_lakewash', 'view_lakesamm', 'view_otherwater', 
                         'view_other']:
        PREDEFINED_BINS[view_feature] = [-np.inf, -3, -1, 0, 1, 3, np.inf]
        
    
    # combined_feature_diffs = np.concatenate(feature_values_list)
    # _, bin_edges = pd.cut(combined_feature_diffs, bins=bins, retbins=True)
    # return bin_edges

    if feature_name in PREDEFINED_BINS:
        bin_edges = PREDEFINED_BINS[feature_name]
    else:
        combined_feature_diffs = np.concatenate(feature_values_list)
        _, bin_edges = pd.cut(combined_feature_diffs, bins=bins, retbins=True, include_lowest=True, duplicates='drop')
    return bin_edges
    
# Function to calculate conditional accuracy
def calculate_conditional_spearman(shap_diffs, feature_diffs, bin_edges=None):
    if bin_edges is None:
        categories = np.unique(feature_diffs)
        conditions = categories
    else:
        conditions = pd.cut(feature_diffs, bins=bin_edges, include_lowest=True)
        categories = conditions.categories
    
    spearman_results = []
    for condition in categories:
        if bin_edges is not None:
            condition_indices = (conditions == condition)
        else:
            condition_indices = (feature_diffs == condition)

        print("<<<Condition<<<")
        print(condition)
        
        condition_shap_diffs = shap_diffs[condition_indices]
        condition_feature_diffs = feature_diffs[condition_indices]
        print(len(condition_shap_diffs), len(condition_feature_diffs))
        
        if len(condition_shap_diffs) > 1:  # Ensure there are enough data points for correlation
            spearman_corr, _ = spearmanr(condition_shap_diffs, condition_feature_diffs)
        else:
            spearman_corr = np.nan  # Not enough data points, set correlation to NaN
        
        count = np.sum(condition_indices)
        
        spearman_results.append({
            'category': str(condition),
            'spearman_corr': spearman_corr,
            'count': count
        })
    
    return pd.DataFrame(spearman_results)

def calculate_conditional_spearman_sim(shap_diffs, feature_diffs, sim_scores, bins=5):
    """
    Calculate conditional accuracy using Spearman rank correlation, conditioned on similarity scores.

    Parameters
    ----------
    shap_diffs : np.ndarray
        Array of SHAP value differences.
    feature_diffs : np.ndarray
        Array of feature differences.
    sim_scores : np.ndarray
        Similarity scores for conditioning.
    bins : int, optional [DEPRECATE]
        Number of bins to use for conditioning on similarity scores. Default is 5.

    Returns
    -------
    spearman_results : pd.DataFrame
        DataFrame containing conditional Spearman rank correlations.
    """
    bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Default bin edges
    
    # Create bins for similarity scores
    rounded_sim_scores = np.round(sim_scores, 2)  # Round to two digit after decimal
    conditions = pd.cut(rounded_sim_scores, bins=bin_edges, include_lowest=True)
    categories = conditions.categories
    spearman_results = []
    for condition in categories:
        condition_indices = (conditions == condition)

        print("<<<Condition<<<")
        print(condition)
        
        condition_shap_diffs = shap_diffs[condition_indices]
        condition_feature_diffs = feature_diffs[condition_indices]
        print("#data: ",len(condition_shap_diffs), len(condition_feature_diffs))
        
        if len(condition_shap_diffs) > 1:  # Ensure there are enough data points for correlation
            spearman_corr, _ = spearmanr(condition_shap_diffs, condition_feature_diffs)
        else:
            spearman_corr = np.nan  # Not enough data points, set correlation to NaN
        
        count = np.sum(condition_indices)
        
        spearman_results.append({
            'category': str(condition),
            'spearman_corr': spearman_corr,
            'count': count
        })
    
    return pd.DataFrame(spearman_results)
    
def compute_monotonicity_with_diffs(shap_values_list, feature_values_list, sim_scores_list, method_names, feature_names, output_csv):
    """
    Compute monotonicity between SHAP differences and feature differences using Spearman rank correlation
    and the counts and ratios of various (shap_diff, feature_diff) combinations.

    Parameters
    ----------
    shap_values_list : list of np.ndarray
        List of SHAP value difference arrays from different methods.
    feature_values_list : list of np.ndarray
        List of feature difference arrays from different methods.
    sim_scores_list: list of dict of np.ndarray
    
    method_names : list of str
        The names of the methods used to calculate SHAP values.
    feature_names : list of str
        List of feature names.
    output_csv : str
        Path to save the CSV file with monotonicity results.

    Returns
    -------
    monotonicity_results : dict
        A dictionary containing monotonicity results for each method and each feature.
    """
    num_methods = len(shap_values_list)
    num_features = shap_values_list[0].shape[1]
    monotonicity_results = {method: {} for method in method_names}
    csv_data = []

    conditional_spearman_dfs = []
    conditional_spearman_sim_dfs = []

    # List of features that need binning
    features_to_bin = feature_names
    # Calculate unified bins for each feature that needs binning
    unified_bins = {}
    for feature_idx, feature_name in enumerate(feature_names):
        if feature_name in features_to_bin:
            feature_values_all_methods = [features[:, feature_idx] for features in feature_values_list]
            unified_bins[feature_name] = get_unified_bins(feature_name, feature_values_all_methods, bins=5)

    for method_idx in range(num_methods):
        method = method_names[method_idx]
        shap_values = shap_values_list[method_idx]
        feature_values = feature_values_list[method_idx]
        sim_scores_dict = sim_scores_list[method_idx]

        for feature_idx in range(num_features):
            shap_diffs = shap_values[:, feature_idx]
            feature_diffs = feature_values[:, feature_idx]
            feature_name = feature_names[feature_idx]

            # Calculate Spearman correlation as a measure of monotonicity
            spearman_corr, _ = spearmanr(shap_diffs, feature_diffs)

            # Calculate the counts and ratios for all combinations of shap_diffs and feature_diffs
            status_counts, status_ratios = compute_status_counts(shap_diffs, feature_diffs)
            
            result = {
                'method': method, 
                'feature': feature_names[feature_idx], 
                'spearman_corr': spearman_corr,
                '(+,+)_count': status_counts['(+,+)'], '(+,+)_ratio': status_ratios['(+,+)'],
                '(+,-)_count': status_counts['(+,-)'], '(+,-)_ratio': status_ratios['(+,-)'],
                '(+,0)_count': status_counts['(+,0)'], '(+,0)_ratio': status_ratios['(+,0)'],
                '(-,+)_count': status_counts['(-,+)'], '(-,+)_ratio': status_ratios['(-,+)'],
                '(-,-)_count': status_counts['(-,-)'], '(-,-)_ratio': status_ratios['(-,-)'],
                '(-,0)_count': status_counts['(-,0)'], '(-,0)_ratio': status_ratios['(-,0)'],
                '(0,+)_count': status_counts['(0,+)'], '(0,+)_ratio': status_ratios['(0,+)'],
                '(0,-)_count': status_counts['(0,-)'], '(0,-)_ratio': status_ratios['(0,-)'],
                '(0,0)_count': status_counts['(0,0)'], '(0,0)_ratio': status_ratios['(0,0)']
            }

            monotonicity_results[method][feature_names[feature_idx]] = result
            
            # Add to CSV data
            csv_data.append(result)
            
            # Calculate conditional Spearman rank correlation [Conditioned on feature value diffs, not the original feature values]
            if feature_name in ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt',
                                'garb_sqft', 'gara_sqft',
                                'days_since_sale', 'home_age', 'reno_age', 
                                'location_value']:  # Numerical features
                print(f">>>>>>>>{feature_name}>>>>>>>>>")
                spearman_df = calculate_conditional_spearman(shap_diffs, feature_diffs, bin_edges=unified_bins[feature_name])
            else:  # Categorical features, maybe also cut on bins
                print(f">>>>>>>>{feature_name}>>>>>>>>>")
                spearman_df = calculate_conditional_spearman(shap_diffs, feature_diffs, bin_edges=unified_bins[feature_name])
            
            spearman_df['feature'] = feature_name
            spearman_df['method'] = method
            conditional_spearman_dfs.append(spearman_df)

            # calculate sim scores conditions:
            if method == 'Pairwise-comps':
                sim_scores_dict['comps_sim_scores_rev'] = 1 - np.array(sim_scores_dict['comps_sim_scores'])
                spearman_sim_df = calculate_conditional_spearman_sim(shap_diffs, feature_diffs, 
                                                                     sim_scores_dict['comps_sim_scores_rev']) #NOTE: comps_sim_scores is transformed into "1 - comps_sim_scores"
            else:
                spearman_sim_df = calculate_conditional_spearman_sim(shap_diffs, feature_diffs, 
                                                                     sim_scores_dict['cos_sim_scores'])
            spearman_sim_df['feature'] = feature_name
            spearman_sim_df['method'] = method
            conditional_spearman_sim_dfs.append(spearman_sim_df)
            

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv, index=False)
    #### Add some plots for the monotonicity df
    plot_monotonicity(df, os.path.dirname(output_csv))

    # # Print results in a structured way
    # print("Monotonicity Check Results:")
    # for method, features in monotonicity_results.items():
    #     print(f"\nMethod: {method}")
    #     table = []
    #     for feature, result in features.items():
    #         table.append([feature, result['spearman_corr'], result['pos_neg_ratio']])
    #     print(tabulate(table, headers=["Feature", "Spearman Correlation", "Pos/Neg Ratio"], tablefmt="pretty"))
        
    # Combine conditional Spearman DataFrames and plot
    combined_conditional_spearman_df = pd.concat(conditional_spearman_dfs)
    plot_conditional_accuracy_heatmap(combined_conditional_spearman_df, os.path.dirname(output_csv))

    # Combine conditional Spearman - Sim DataFrames and plot
    combined_conditional_spearman_sim_df = pd.concat(conditional_spearman_sim_dfs)
    combined_conditional_spearman_sim_df.to_csv(os.path.join(os.path.dirname(output_csv), "conditional_spearman_sim.csv"), index=False)
    
    plot_conditional_accuracy_heatmap_sim(combined_conditional_spearman_sim_df, os.path.dirname(output_csv))

    return monotonicity_results



if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    version = args.model_version

    # full set of exp methods:
    method_names = ["Marginal-all","Marginal-kmeans","Baseline-0","Baseline-median","Uniform","TreeShap-treepath", "Conditional-all"]
    method_names_pair = ['Pairwise-random', 'Pairwise-comps', 'Pairwise-sim'] 
    base_model = f'base_model_{version}'

    if dataset == "kingcounty":
        if version != "v3":
            method_names.remove("Conditional-all")

    else:
        raise ValueError("post exp analysis only supports king county dataset!")
        # method_names.remove("Conditional-all")
        # method_names_pair.remove("Pairwise-comps")
    
    shap_paths = list(map(lambda method: f'../results/{dataset}/{base_model}/{method}/shap_values.pkl', method_names))
    shap_paths_pair = list(map(lambda method: f'../results/{dataset}/{base_model}/{method}/shap_values.pkl', method_names_pair))
    
    output_dir = f'../results/{dataset}/{base_model}/post_exp_results'
    os.makedirs(output_dir, exist_ok=True)
    
    sim_paths_pair = list(map(lambda method: f'../results/{dataset}/{base_model}/{method}/pairwise_similarity_scores.pkl', method_names_pair))
    
    # setup logging
    setup_logging(results_dir = output_dir, filename = 'post_exlain.log')
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"Explaining for methods: {method_names+method_names_pair}")
    
    feature_names = pd.read_csv(f'../results/{dataset}/{base_model}/X_test.csv', nrows=0).columns.tolist()
        
    # print(f"Considering features: {feature_names}")
    # Load SHAP values
    shap_values_list, feature_values = load_shap_values(shap_paths)
    shap_values_list_pair, feature_values_pair = load_shap_values_pair(shap_paths_pair)
    
    # Load sim scores for pairwise methods
    sim_scores_list_of_dict_pair = load_sim_scores_pair(sim_paths_pair)
    
    
    # Plot the distribution
    plot_shap_distributions(shap_values_list+shap_values_list_pair, feature_names, method_names+method_names_pair, output_dir)
    # plot_shap_distributions(shap_values_list_pair, feature_names, method_names_pair, output_dir)
    
    # normalized_diffs = plot_normalized_distribution(shap_values_list, feature_values, feature_names, method_names, output_dir)
    normalized_diffs, shap_values_list_diff, feature_values_list_diff, similarity_scores_list_of_dict_nonpair = plot_normalized_distribution(shap_values_list, feature_values,
                                                    shap_values_list_pair, feature_values_pair,
                                                    feature_names, method_names, method_names_pair, output_dir)

    # Monotonicity Check
    monotonicity_results = compute_monotonicity_with_diffs(shap_values_list_diff + shap_values_list_pair, 
                                                           feature_values_list_diff + feature_values_pair, 
                                                           similarity_scores_list_of_dict_nonpair + sim_scores_list_of_dict_pair,
                                                           method_names + method_names_pair, 
                                                           feature_names, 
                                                           output_csv=os.path.join(output_dir, 'monotonicity_results.csv'))
    print("Monotonicity Results:", monotonicity_results)
    # Compute KL divergence and correlation
    kl_divergences, correlations = compute_kl_divergence_and_correlation(shap_values_list, normalized_diffs, feature_values, method_names, symmetry=True)
    
    ## Plot KL divergence and correlation heatmaps
    # plot_kl_and_correlation_heatmaps(kl_divergences, correlations, feature_names, method_names, output_dir)

    # SHAP dist'n vs corr
    threshold = 0.3 # threshold for "highly" correlated features
    analyze_shap_distributions_wcorr(correlations, shap_values_list+shap_values_list_pair, normalized_diffs, feature_names, method_names+method_names_pair, threshold, output_dir)

    



















