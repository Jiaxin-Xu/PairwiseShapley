import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import shap
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
import joblib
import os
import random
from utils import setup_logging, set_global_random_seed, transform_shapr_to_shap
import argparse
import warnings
import time
import pickle
import shaprpy
from tpot import TPOTRegressor

from pairwise_shapley import pairwise_shapley_explanation
from comparable_candidates import get_comparable_candidate_indices
from similar_candidates import get_similarity_based_candidate_indices, cosine_similarity_scores, euclidean_distance_scores

import matplotlib.cm as cm


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Post-hoc explaination on model for King County home price (and two more molecule/material datasets) prediction.')
    parser.add_argument('--explain_method', type=str, default='Conditional', 
                        choices=['Marginal-all', 
                                 'Marginal-kmeans', 
                                 'Conditional-all',
                                 'Baseline-0', 
                                 'Baseline-median',
                                 'Uniform', 
                                 'TreeShap-treepath', 
                                 'TreeShap-interventional',
                                 'Pairwise-random', 'Pairwise-comps', 'Pairwise-sim', 'Pairwise'
                                ], 
                        help='Method to explain the ML model prediction')

    # parser.add_argument('--model_version', type=str, default='9', 
    #                     choices=['9', '8', '7', '10'
    #                             ], 
    #                     help='base model version')

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
    
    parser.add_argument('--num_pairs', type=int, default=50,
                        help='Number of pairs for pairwise-shapley, as well as for later non-pairwise method normalization calculation')
    
    return parser.parse_args()

class ModelWrapper:
    """
    A wrapper for machine learning models to unify the predict method for both
    classification (binary) and regression tasks.

    Attributes
    ----------
    model : object
        The underlying machine learning model.
    is_classification : bool
        A flag indicating whether the model is a classification model.

    Methods
    -------
    predict(X)
        Predicts the output for the given input data X. For classification models,
        it returns the probabilities for the positive class.
    __getattr__(name)
        Forwards attribute access to the underlying model.
    """

    def __init__(self, model, is_classification=False):
        """
        Initializes the ModelWrapper with the given model and classification flag.

        Parameters
        ----------
        model : object
            The machine learning model to be wrapped.
        is_classification : bool, optional
            Indicates if the model is a classification model (default is False).
        """
        self.model = model
        self.is_classification = is_classification

    def predict(self, X):
        """
        Predicts the output for the given input data X.

        For classification models, it returns the probabilities for the positive class.

        Parameters
        ----------
        X : array-like
            Input data to predict.

        Returns
        -------
        array
            Predicted output. For classification models, returns probabilities for the positive class.
        """
        if self.is_classification:
            probabilities = self.model.predict_proba(X)
            # Return the probabilities for the first class
            return probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        else:
            return self.model.predict(X)

    # def __getattr__(self, name):
    #     """
    #     Forwards attribute access to the underlying model.

    #     This allows the wrapper to behave as transparently as possible,
    #     forwarding calls for methods and attributes to the underlying model.

    #     Parameters
    #     ----------
    #     name : str
    #         Name of the attribute to access.

    #     Returns
    #     -------
    #     attribute
    #         The requested attribute from the underlying model.
    #     """
    #     try:
    #         # Try to get the attribute from the current instance
    #         return self.__getattribute__(name)
    #     except AttributeError:
    #         # If the attribute is not found, try to get it from the underlying model
    #         try:
    #             return getattr(self.model, name)
    #         except AttributeError:
    #             # Raise an error if the attribute is not found in both
    #             raise AttributeError(f"'{type(self).__name__}' object and its wrapped model have no attribute '{name}'")
        
def get_final_model(model):
    """
    Get the final estimator from the pipeline or the model itself if it's not a pipeline.
    Args:
        model: The machine learning model or pipeline.
    
    Returns:
        The final model (e.g., the classifier/regressor itself).
    """
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model

def preprocess_data(pipeline, X):
    """
    Apply the preprocessing steps of the pipeline to the data.
    Args:
        pipeline: The pipeline containing preprocessing steps.
        X: The input data to preprocess.
    
    Returns:
        The preprocessed data.
    """
    if isinstance(pipeline, Pipeline):
        X_transformed = X
        # Apply all steps except the final model step
        for name, step in pipeline.steps[:-1]:
            X_transformed = step.transform(X_transformed)
        return X_transformed
    return X

def is_tree_based(model):
    """
    Check if the model is tree-based (e.g., decision tree, random forest, gradient boosting).
    Args:
        model: The machine learning model or pipeline.
    
    Returns:
        bool: True if the model is tree-based, False otherwise.
    """
    tree_based_models = (
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "RandomForestClassifier", "RandomForestRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "XGBClassifier", "XGBRegressor",
        "LGBMClassifier", "LGBMRegressor",
        "CatBoostClassifier", "CatBoostRegressor"
    )
    final_model = get_final_model(model)
    
    return final_model.__class__.__name__ in tree_based_models


def explain_single(results_dir, dataset, explain_method, model_version, n_pairs, single_id=0):
    """
    For computation time comparison only
    """

    # Create results_dir if it does not exist
    results_dir_exp = os.path.join(results_dir, explain_method)
    os.makedirs(results_dir_exp, exist_ok=True)
    # setup logging
    setup_logging(results_dir = results_dir_exp, filename = 'exlain_singledata.log')
    # setup warning filter
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # set global random seed
    set_global_random_seed(42)

    # read the saved data
    X_train_val_combined = pd.read_csv(os.path.join(results_dir, 'X_train_val_combined.csv'))
    X_test = pd.read_csv(os.path.join(results_dir, 'X_test.csv'))

    if dataset == "kingcounty":
        y_test_pred = np.load(os.path.join(results_dir, 'y_test_pred.npy'))
        y_train_pred = pd.Series(np.load(os.path.join(results_dir, 'y_train_pred.npy')))
        y_val_pred = pd.Series(np.load(os.path.join(results_dir, 'y_val_pred.npy')))
        y_tr_val_pred_combined = pd.concat([y_train_pred, y_val_pred])
    elif dataset == "polymer":
        y_test_pred = np.load(os.path.join(results_dir, 'y_test_pred.npy'))
        y_train_pred = pd.Series(np.load(os.path.join(results_dir, 'y_train_pred.npy')))
        y_tr_val_pred_combined = y_train_pred
    else: # hiv dataset -- use predicted probability of class 1
        y_test_pred = np.load(os.path.join(results_dir, 'y_test_pred_prob.npy'))[:, 1]
        y_train_pred = pd.Series(np.load(os.path.join(results_dir, 'y_train_pred_prob.npy'))[:, 1])
        y_tr_val_pred_combined = y_train_pred
        

    # load the trained TPOT pipeline
    pipeline = joblib.load(os.path.join(results_dir, 'tpot_pipeline.pkl'))

    # transform the data if the pipeline has preprocessing steps
    X_train_val_transformed = preprocess_data(pipeline, X_train_val_combined)
    X_test_transformed = preprocess_data(pipeline, X_test)
    X_test_transformed = X_test_transformed.iloc[[single_id],:] # only explain the first data!
    
    # get the final model from the pipeline
    final_model = get_final_model(pipeline)

    ## get explain method
    warnings.filterwarnings("ignore", category=UserWarning)
    # Start time measurement
    start_time = time.time()

    ####### EXPLAIN TYPE I: Baseline #########
    if explain_method == "Baseline-median":
        explainer = shap.KernelExplainer(final_model.predict,
                                         X_train_val_combined.median().values.reshape((1, X_train_val_combined.shape[1])),
                                         feature_names = X_train_val_combined.columns.to_list())
        print(f"Using KernelExplainer [Baseline-median] for model: {final_model.__class__.__name__}")
    elif explain_method == "Baseline-0":
        explainer = shap.KernelExplainer(final_model.predict,
                                         np.zeros((1, X_train_val_combined.shape[1])),
                                         feature_names = X_train_val_combined.columns.to_list())
        print(f"Using KernelExplainer [Baseline-0] for model: {final_model.__class__.__name__}")

    ####### EXPLAIN TYPE II: Uniform #########
    elif explain_method == "Uniform": 
        explainer = shap.SamplingExplainer(final_model.predict,
                                         X_train_val_combined.sample(n=100, random_state=42),
                                         feature_names = X_train_val_combined.columns.to_list())
        print(f"Using SamplingExplainer [Uniform] for model: {final_model.__class__.__name__}")
    
     ## KenerlShap are marginal!!! (independent features imputation[sampling from background]) 
     ####### EXPLAIN TYPE III: Marginal #########
    elif explain_method == "Marginal-all": 
        explainer = shap.KernelExplainer(final_model.predict,
                                         X_train_val_combined.sample(n=100, random_state=42), #change from 500 to 100 on v7 and v9
                                         feature_names = X_train_val_combined.columns.to_list())
        print(f"Using KernelExplainer [Marginal-All] for model: {final_model.__class__.__name__}")
    elif explain_method == "Marginal-kmeans": 
        # rather than use the whole training set to estimate expected values, we summarize with
        # a set of weighted kmeans, each weighted by the number of points they represent.
        X_train_val_combined_summary = shap.kmeans(X_train_val_combined, 10)
        explainer = shap.KernelExplainer(final_model.predict,
                                         X_train_val_combined_summary,
                                         feature_names = X_train_val_combined.columns.to_list())
        print(f"Using KernelExplainer [Conditional-Kmeans10] for model: {final_model.__class__.__name__}")
    ####### EXPLAIN TYPE IV: Conditional #########
    elif explain_method == "Conditional-all":
        if dataset != "kingcounty":
            raise ValueError("Conditional-all currently only supports KingCounty Dataset (<=30 features version)!")
        # use implementation of shapr: https://github.com/NorskRegnesentral/shapr.git
        # Explain the model using the empirical approach
        explainer = shaprpy.explain(
            model=final_model,
            x_train=X_train_val_combined.sample(n=100, random_state=42),
            x_explain=X_test_transformed,
            approach='empirical',
            prediction_zero=y_tr_val_pred_combined.mean().item(),
            n_combinations = 100,
            n_batches = 10
        )
        print(f"Using shaprpy-Conditional-shap for model: {final_model.__class__.__name__}")
        
    ####### EXPLAIN TYPE V: Model-specific [Tree] #########
    elif explain_method == "TreeShap-treepath":
        if is_tree_based(final_model):
            explainer = shap.TreeExplainer(final_model, feature_perturbation='tree_path_dependent')
            print(f"Using TreeExplainer for model: {final_model.__class__.__name__}")
        else:
            raise TypeError("The final_model from TPOT is not Tree-based model, thus cannot be explained by TreeSHAP!")
    
    elif explain_method == "TreeShap-interventional":
        if is_tree_based(final_model):
            explainer = shap.TreeExplainer(final_model, data = X_train_val_combined.sample(n=100, random_state=42), feature_perturbation='interventional')
            print(f"Using TreeExplainer for model: {final_model.__class__.__name__}")
        else:
            raise TypeError("The final_model from TPOT is not Tree-based model, thus cannot be explained by TreeSHAP!")
    
    ####### EXPLAIN TYPE VI: Pairwise #########
    elif explain_method == "Pairwise-random":  
        explicands = X_test_transformed.to_numpy()
        candidates = X_train_val_combined.to_numpy()
        # Generates a list where each element is a list of n_pairs unique random indices from the candidates array, corresponding to each explicand.
        candidate_indices = [np.random.choice(range(candidates.shape[0]), n_pairs, replace=False) for _ in range(explicands.shape[0])]
        start_time_pair = time.time()
        pairwise_shap_values, _ = pairwise_shapley_explanation(
            final_model, dataset, explicands, candidates, candidate_indices,
            feature_names=X_train_val_combined.columns.to_list(),
            compute_sim=False)
    
    elif explain_method == "Pairwise-comps":
        if dataset != "kingcounty":
            raise ValueError("Pairwise-comps currently only supports KingCounty Dataset! (Comps algorithm for new datasets needs to be defined specifically)")
        candidate_indices, candidate_sim_scores = get_comparable_candidate_indices(
            base_model_dir = results_dir,
            max_radius=2, k=n_pairs, top_n=n_pairs
        )
        explicands = X_test_transformed.to_numpy()
        candidates = X_train_val_combined.to_numpy()
        start_time_pair = time.time()
        pairwise_shap_values, _ = pairwise_shapley_explanation(
            final_model, dataset, explicands, candidates, candidate_indices,
            feature_names=X_train_val_combined.columns.to_list(),
            compute_sim=False)
        
    elif explain_method == "Pairwise-sim":
        if dataset == "kingcounty":
            if model_version == "v1":
                # no days_since_sale
                sim_feature_names = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 
                                     'grade', 'fbsmt_grade', 'condition', 
                                     'stories', 'beds', 'bath_comb', 
                                     'garb_sqft', 'gara_sqft', 
                                     'wfnt', 'golf', 'greenbelt', 'noise_traffic', 
                                     'view_rainier', 'view_olympics', 'view_cascades',
                                     'view_territorial', 'view_skyline', 'view_sound', 
                                     'view_lakewash', 'view_lakesamm', 'view_otherwater', 
                                     'view_other', 
                                     'home_age', 'reno_age', 'location_value', 
                                     'city_BELLEVUE', 'city_FEDERAL WAY', 'city_KENT', 
                                     'city_KING COUNTY', 'city_KIRKLAND', 'city_RENTON', 
                                     'city_SAMMAMISH', 'city_SEATTLE', 'city_SHORELINE', 
                                     'city_other', 'submarket_D', 'submarket_F', 
                                     'submarket_I', 'submarket_K', 'submarket_L', 
                                     'submarket_M', 'submarket_O', 'submarket_Q', 
                                     'submarket_R', 'submarket_other']
            elif model_version == "v0":
                # no days_since_sale
                sim_feature_names = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 
                                     'grade', 'fbsmt_grade', 'condition', 
                                     'stories', 'beds', 'bath_full','bath_3qtr','bath_half', 
                                     'garb_sqft', 'gara_sqft', 
                                     'wfnt', 'golf', 'greenbelt', 'noise_traffic', 
                                     'view_rainier', 'view_olympics', 'view_cascades',
                                     'view_territorial', 'view_skyline', 'view_sound', 
                                     'view_lakewash', 'view_lakesamm', 'view_otherwater', 
                                     'view_other', 
                                     'home_age', 'reno_age', 'location_value', 
                                     'city_BELLEVUE', 'city_FEDERAL WAY', 'city_KENT', 
                                     'city_KING COUNTY', 'city_KIRKLAND', 'city_RENTON', 
                                     'city_SAMMAMISH', 'city_SEATTLE', 'city_SHORELINE', 
                                     'city_other', 'submarket_D', 'submarket_F', 
                                     'submarket_I', 'submarket_K', 'submarket_L', 
                                     'submarket_M', 'submarket_O', 'submarket_Q', 
                                     'submarket_R', 'submarket_other']
            elif model_version == "v2":
                # no days_since_sale
                sim_feature_names = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 'grade', 'fbsmt_grade', 
                                     'condition', 'stories', 'beds', 'bath_comb', 'garb_sqft', 'gara_sqft', 
                                     'wfnt', 'noise_traffic', 'view_olympics', 
                                     'view_cascades', 'view_territorial', 'view_skyline', 'view_sound', 'view_lakewash', 
                                     'view_lakesamm', 
                                     'home_age', 'reno_age', 'location_value', 'city_BELLEVUE', 'city_FEDERAL WAY', 
                                     'city_KENT', 'city_KING COUNTY', 'city_KIRKLAND', 'city_RENTON', 'city_SAMMAMISH', 
                                     'city_SEATTLE', 'city_SHORELINE', 'city_other', 'submarket_D', 'submarket_F', 'submarket_I', 
                                     'submarket_K', 'submarket_L', 'submarket_M', 'submarket_O', 'submarket_Q', 'submarket_R', 'submarket_other']
            elif model_version == "v3":
                # no days_since_sale
                sim_feature_names = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 'grade',
                                     'condition', 'stories', 'beds', 'bath_comb',
                                     'wfnt', 'noise_traffic', 'view_olympics', 
                                     'view_sound', 'view_lakewash', 
                                     'view_lakesamm', 
                                     'home_age', 'reno_age', 'location_value', 'city_BELLEVUE', 
                                     'city_KIRKLAND', 'city_SEATTLE', 'city_other', 'submarket_D', 'submarket_I', 
                                     'submarket_M', 'submarket_O', 'submarket_Q', 'submarket_R', 'submarket_other']
        
            else:
                raise ValueError(f"Unexpected model_version for king county dataset: {model_version}")
        else: # for other datasets, using all features
            sim_feature_names = X_test.columns.tolist()
            
        similarity_func = cosine_similarity_scores
        candidate_indices = get_similarity_based_candidate_indices(
            base_model_dir = results_dir,
            dataset = dataset,
            sim_feature_names=sim_feature_names,
            k=n_pairs,
            similarity_func=similarity_func
        )
        explicands = X_test_transformed.to_numpy()
        candidates = X_train_val_combined.to_numpy()
        start_time_pair = time.time()
        pairwise_shap_values, _ = pairwise_shapley_explanation(
            final_model, dataset, explicands, candidates, candidate_indices,
            feature_names=X_train_val_combined.columns.to_list(),
            compute_sim=False)
        
    else:
        raise ValueError("Unsupported explanation method!")
    
    #########################################################
    #########################################################
    #########################################################
    #########################################################
    ## calculate shap values
    if explain_method == "Conditional-all":
        # print(explainer)
        shap_values = transform_shapr_to_shap(explainer, X_test_transformed, feature_names=X_train_val_combined.columns.to_list())
    elif explain_method in ["Pairwise-random", "Pairwise-comps", "Pairwise-sim"]:
        shap_values = pairwise_shap_values  # For compatibility
    else:
        shap_values = explainer(X_test_transformed)
    
    # End time measurement
    end_time = time.time()
    if explain_method in ["Pairwise-random", "Pairwise-comps", "Pairwise-sim"]:
        elapsed_time_pair = end_time - start_time_pair
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(f"Total running time (excl. comps.) of {explain_method}: {elapsed_time_pair:.2f} seconds")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        return elapsed_time_pair
    elapsed_time = end_time - start_time     
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"Total running time of {explain_method}: {elapsed_time:.2f} seconds")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return elapsed_time
    

## kernel shap sends data as numpy array which has no column names, so we fix it
def xgb_predict(estimator, feature_names, data_asarray):
    data_asframe =  pd.DataFrame(data_asarray, columns=feature_names)
    return estimator.predict(data_asframe)

def explain(results_dir, dataset, explain_method, model_version, n_pairs):

    # Create results_dir if it does not exist
    results_dir_exp = os.path.join(results_dir, explain_method)
    os.makedirs(results_dir_exp, exist_ok=True)
    # setup logging
    setup_logging(results_dir = results_dir_exp, filename = 'exlain.log')
    # setup warning filter
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # set global random seed
    set_global_random_seed(42)

    # read the saved data
    X_train_val_combined = pd.read_csv(os.path.join(results_dir, 'X_train_val_combined.csv'))
    X_test = pd.read_csv(os.path.join(results_dir, 'X_test.csv'))
    if dataset == "kingcounty":
        y_test_pred = np.load(os.path.join(results_dir, 'y_test_pred.npy'))
        y_train_pred = pd.Series(np.load(os.path.join(results_dir, 'y_train_pred.npy')))
        y_val_pred = pd.Series(np.load(os.path.join(results_dir, 'y_val_pred.npy')))
        y_tr_val_pred_combined = pd.concat([y_train_pred, y_val_pred])
    elif dataset == "polymer":
        y_test_pred = np.load(os.path.join(results_dir, 'y_test_pred.npy'))
        y_train_pred = pd.Series(np.load(os.path.join(results_dir, 'y_train_pred.npy')))
        y_tr_val_pred_combined = y_train_pred
    else: # hiv dataset -- use predicted probability of class 1
        y_test_pred = np.load(os.path.join(results_dir, 'y_test_pred_prob.npy'))[:, 1]
        y_train_pred = pd.Series(np.load(os.path.join(results_dir, 'y_train_pred_prob.npy'))[:, 1])
        y_tr_val_pred_combined = y_train_pred

    # load the trained TPOT pipeline
    pipeline = joblib.load(os.path.join(results_dir, 'tpot_pipeline.pkl'))

    # transform the data if the pipeline has preprocessing steps
    X_train_val_transformed = preprocess_data(pipeline, X_train_val_combined)
    X_test_transformed = preprocess_data(pipeline, X_test)
    if dataset == "hiv": # hiv dataset has too many test data, reduce the number to save time for explanation
        X_test_transformed = X_test_transformed.iloc[:500,:]
    # ######## TODO!!! For now, all the tpot models have no preprocessing steps
    # X_train_val_transformed = X_train_val_combined
    # X_test_transformed = X_test
    
    # get the final model from the pipeline
    final_model = get_final_model(pipeline)
    
    # if dataset == "hiv": #classification
    #     final_model_predict = final_model.predict_proba
    # else: #regression
    #     final_model_predict = final_model.predict

    if dataset == "hiv":  # classification
        final_model_wrapper = ModelWrapper(final_model, is_classification=True)
    else:  # regression
        final_model_wrapper = ModelWrapper(final_model, is_classification=False)
    
    clust = shap.utils.hclust(X_test, y_test_pred, linkage="single")

    ## get explain method
    warnings.filterwarnings("ignore", category=UserWarning)
    # Start time measurement
    start_time = time.time()

    ####### EXPLAIN TYPE I: Baseline #########
    if explain_method == "Baseline-median":
        explainer = shap.KernelExplainer(final_model_wrapper.predict,#final_model.predict,
                                         X_train_val_combined.median().values.reshape((1, X_train_val_combined.shape[1])),
                                         feature_names = X_train_val_combined.columns.to_list())
        print(f"Using KernelExplainer [Baseline-median] for model: {final_model.__class__.__name__}")
    elif explain_method == "Baseline-0":
        explainer = shap.KernelExplainer(final_model_wrapper.predict,#final_model.predict,
                                         np.zeros((1, X_train_val_combined.shape[1])),
                                         feature_names = X_train_val_combined.columns.to_list())
        print(f"Using KernelExplainer [Baseline-0] for model: {final_model.__class__.__name__}")

    ####### EXPLAIN TYPE II: Uniform #########
    elif explain_method == "Uniform": 
        explainer = shap.SamplingExplainer(final_model_wrapper.predict,#final_model.predict,
                                         X_train_val_combined.sample(n=100, random_state=42),
                                         # X_train_val_combined.sample(n=100, random_state=42),
                                         feature_names = X_train_val_combined.columns.to_list())
        print(f"Using SamplingExplainer [Uniform] for model: {final_model.__class__.__name__}")
    
     ## KenerlShap are marginal!!! (independent features imputation[sampling from background]) 
     ####### EXPLAIN TYPE III: Marginal #########
    elif explain_method == "Marginal-all": 
        explainer = shap.KernelExplainer(final_model_wrapper.predict,#final_model.predict,
                                         X_train_val_combined.sample(n=100, random_state=42), #change from 500 to 100 on v7 and v9
                                         feature_names = X_train_val_combined.columns.to_list())
        print(f"Using KernelExplainer [Marginal-All] for model: {final_model.__class__.__name__}")
    elif explain_method == "Marginal-kmeans": 
        # rather than use the whole training set to estimate expected values, we summarize with
        # a set of weighted kmeans, each weighted by the number of points they represent.
        X_train_val_combined_summary = shap.kmeans(X_train_val_combined, 10)
        explainer = shap.KernelExplainer(final_model_wrapper.predict,#final_model.predict,
                                         X_train_val_combined_summary,
                                         feature_names = X_train_val_combined.columns.to_list())
        print(f"Using KernelExplainer [Marginal-Kmeans10] for model: {final_model.__class__.__name__}")
    ####### EXPLAIN TYPE IV: Conditional #########
    elif explain_method == "Conditional-all":    
        if dataset != "kingcounty":
            raise ValueError("Conditional-all currently only supports KingCounty Dataset (<=30 features version)!")
        # use implementation of shapr: https://github.com/NorskRegnesentral/shapr.git
        # Explain the model using the empirical approach
        explainer = shaprpy.explain(
            model=final_model_wrapper, 
            x_train=X_train_val_combined.sample(n=100, random_state=42),
            x_explain=X_test_transformed,
            approach='empirical',
            prediction_zero=y_tr_val_pred_combined.mean().item(),
            n_combinations = 100,
            n_batches = 10
        )
        print(f"Using shaprpy-Conditional-shap for model: {final_model.__class__.__name__}")
        
    ####### EXPLAIN TYPE V: Model-specific [Tree] #########
    elif explain_method == "TreeShap-treepath":
        if is_tree_based(final_model):
            explainer = shap.TreeExplainer(final_model_wrapper, #final_model, 
                                           feature_perturbation='tree_path_dependent')
            print(f"Using TreeExplainer for model: {final_model.__class__.__name__}")
        else:
            raise TypeError("The final_model from TPOT is not Tree-based model, thus cannot be explained by TreeSHAP!")
    
    elif explain_method == "TreeShap-interventional":
        if is_tree_based(final_model):
            explainer = shap.TreeExplainer(final_model_wrapper, #final_model, 
                                           data = X_train_val_combined.sample(n=100, random_state=42), 
                                           feature_perturbation='interventional')
            print(f"Using TreeExplainer for model: {final_model.__class__.__name__}")
        else:
            raise TypeError("The final_model from TPOT is not Tree-based model, thus cannot be explained by TreeSHAP!")
    
    ####### EXPLAIN TYPE VI: Pairwise #########
    elif explain_method == "Pairwise-random":  
        # explicands = X_test_transformed.to_numpy()
        # candidates = X_train_val_combined.to_numpy()

        try:
            # Ensure X_test_transformed is a pandas DataFrame or Series before calling to_numpy()
            if isinstance(X_test_transformed, (pd.DataFrame, pd.Series)):
                explicands = X_test_transformed.to_numpy()
            elif isinstance(X_test_transformed, np.ndarray):
                # If it's already a numpy array, just use it directly
                explicands = X_test_transformed
            else:
                raise TypeError("Unsupported data type for X_test_transformed")
        
            # Ensure X_train_val_combined is a pandas DataFrame or Series before calling to_numpy()
            if isinstance(X_train_val_combined, (pd.DataFrame, pd.Series)):
                candidates = X_train_val_combined.to_numpy()
            elif isinstance(X_train_val_combined, np.ndarray):
                # If it's already a numpy array, just use it directly
                candidates = X_train_val_combined
            else:
                raise TypeError("Unsupported data type for X_train_val_combined")
        except AttributeError as e:
            print(f"AttributeError: {e}")
        except TypeError as e:
            print(f"TypeError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
        # Generates a list where each element is a list of n_pairs unique random indices from the candidates array, corresponding to each explicand.
        candidate_indices = [np.random.choice(range(candidates.shape[0]), n_pairs, replace=False) for _ in range(explicands.shape[0])]
        start_time_pair = time.time()
        pairwise_shap_values, pairwise_similarity_scores = pairwise_shapley_explanation(
            final_model_wrapper,#final_model, 
            dataset, explicands, candidates, candidate_indices,
            feature_names=X_train_val_combined.columns.to_list(),
            compute_sim=True)
        # shap_values = pairwise_shap_values  # For compatibility
    
    elif explain_method == "Pairwise-comps":
        if dataset != "kingcounty":
            raise ValueError("Pairwise-comps currently only supports KingCounty Dataset! (Comps algorithm for new datasets needs to be defined specifically)")
        candidate_indices, candidate_sim_scores = get_comparable_candidate_indices(
            base_model_dir = results_dir,
            max_radius=2, k=n_pairs, top_n=n_pairs
        )
        # explicands = X_test_transformed.to_numpy()
        # candidates = X_train_val_combined.to_numpy()

        try:
            # Ensure X_test_transformed is a pandas DataFrame or Series before calling to_numpy()
            if isinstance(X_test_transformed, (pd.DataFrame, pd.Series)):
                explicands = X_test_transformed.to_numpy()
            elif isinstance(X_test_transformed, np.ndarray):
                # If it's already a numpy array, just use it directly
                explicands = X_test_transformed
            else:
                raise TypeError("Unsupported data type for X_test_transformed")
        
            # Ensure X_train_val_combined is a pandas DataFrame or Series before calling to_numpy()
            if isinstance(X_train_val_combined, (pd.DataFrame, pd.Series)):
                candidates = X_train_val_combined.to_numpy()
            elif isinstance(X_train_val_combined, np.ndarray):
                # If it's already a numpy array, just use it directly
                candidates = X_train_val_combined
            else:
                raise TypeError("Unsupported data type for X_train_val_combined")
        except AttributeError as e:
            print(f"AttributeError: {e}")
        except TypeError as e:
            print(f"TypeError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
        start_time_pair = time.time()
        pairwise_shap_values, pairwise_similarity_scores = pairwise_shapley_explanation(
            final_model_wrapper,#final_model, 
            dataset, explicands, candidates, candidate_indices,
            feature_names=X_train_val_combined.columns.to_list(),
             compute_sim=True)
        
    elif explain_method == "Pairwise-sim":
        if dataset == "kingcounty":
            if model_version == "v1":
                # no days_since_sale
                sim_feature_names = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 
                                     'grade', 'fbsmt_grade', 'condition', 
                                     'stories', 'beds', 'bath_comb', 
                                     'garb_sqft', 'gara_sqft', 
                                     'wfnt', 'golf', 'greenbelt', 'noise_traffic', 
                                     'view_rainier', 'view_olympics', 'view_cascades',
                                     'view_territorial', 'view_skyline', 'view_sound', 
                                     'view_lakewash', 'view_lakesamm', 'view_otherwater', 
                                     'view_other', 
                                     'home_age', 'reno_age', 'location_value', 
                                     'city_BELLEVUE', 'city_FEDERAL WAY', 'city_KENT', 
                                     'city_KING COUNTY', 'city_KIRKLAND', 'city_RENTON', 
                                     'city_SAMMAMISH', 'city_SEATTLE', 'city_SHORELINE', 
                                     'city_other', 'submarket_D', 'submarket_F', 
                                     'submarket_I', 'submarket_K', 'submarket_L', 
                                     'submarket_M', 'submarket_O', 'submarket_Q', 
                                     'submarket_R', 'submarket_other']
            elif model_version == "v0":
                # no days_since_sale
                sim_feature_names = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 
                                     'grade', 'fbsmt_grade', 'condition', 
                                     'stories', 'beds', 'bath_full','bath_3qtr','bath_half', 
                                     'garb_sqft', 'gara_sqft', 
                                     'wfnt', 'golf', 'greenbelt', 'noise_traffic', 
                                     'view_rainier', 'view_olympics', 'view_cascades',
                                     'view_territorial', 'view_skyline', 'view_sound', 
                                     'view_lakewash', 'view_lakesamm', 'view_otherwater', 
                                     'view_other', 
                                     'home_age', 'reno_age', 'location_value', 
                                     'city_BELLEVUE', 'city_FEDERAL WAY', 'city_KENT', 
                                     'city_KING COUNTY', 'city_KIRKLAND', 'city_RENTON', 
                                     'city_SAMMAMISH', 'city_SEATTLE', 'city_SHORELINE', 
                                     'city_other', 'submarket_D', 'submarket_F', 
                                     'submarket_I', 'submarket_K', 'submarket_L', 
                                     'submarket_M', 'submarket_O', 'submarket_Q', 
                                     'submarket_R', 'submarket_other']
            elif model_version == "v2":
                # no days_since_sale
                sim_feature_names = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 'grade', 'fbsmt_grade', 
                                     'condition', 'stories', 'beds', 'bath_comb', 'garb_sqft', 'gara_sqft', 
                                     'wfnt', 'noise_traffic', 'view_olympics', 
                                     'view_cascades', 'view_territorial', 'view_skyline', 'view_sound', 'view_lakewash', 
                                     'view_lakesamm', 
                                     'home_age', 'reno_age', 'location_value', 'city_BELLEVUE', 'city_FEDERAL WAY', 
                                     'city_KENT', 'city_KING COUNTY', 'city_KIRKLAND', 'city_RENTON', 'city_SAMMAMISH', 
                                     'city_SEATTLE', 'city_SHORELINE', 'city_other', 'submarket_D', 'submarket_F', 'submarket_I', 
                                     'submarket_K', 'submarket_L', 'submarket_M', 'submarket_O', 'submarket_Q', 'submarket_R', 'submarket_other']
            elif model_version == "v3":
                # no days_since_sale
                sim_feature_names = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 'grade',
                                     'condition', 'stories', 'beds', 'bath_comb',
                                     'wfnt', 'noise_traffic', 'view_olympics', 
                                     'view_sound', 'view_lakewash', 
                                     'view_lakesamm', 
                                     'home_age', 'reno_age', 'location_value', 'city_BELLEVUE', 
                                     'city_KIRKLAND', 'city_SEATTLE', 'city_other', 'submarket_D', 'submarket_I', 
                                     'submarket_M', 'submarket_O', 'submarket_Q', 'submarket_R', 'submarket_other']
            else:
                raise ValueError(f"Unexpected model_version: {model_version}")
        else: # for other datasets, using all features
            sim_feature_names = X_test_transformed.columns.tolist()
        
        similarity_func = cosine_similarity_scores
        candidate_indices = get_similarity_based_candidate_indices(
            base_model_dir = results_dir,
            dataset = dataset,
            sim_feature_names=sim_feature_names,
            k=n_pairs,
            similarity_func=similarity_func
        )
        # explicands = X_test_transformed.to_numpy()
        # candidates = X_train_val_combined.to_numpy()
        
        try:
            # Ensure X_test_transformed is a pandas DataFrame or Series before calling to_numpy()
            if isinstance(X_test_transformed, (pd.DataFrame, pd.Series)):
                explicands = X_test_transformed.to_numpy()
            elif isinstance(X_test_transformed, np.ndarray):
                # If it's already a numpy array, just use it directly
                explicands = X_test_transformed
            else:
                raise TypeError("Unsupported data type for X_test_transformed")
        
            # Ensure X_train_val_combined is a pandas DataFrame or Series before calling to_numpy()
            if isinstance(X_train_val_combined, (pd.DataFrame, pd.Series)):
                candidates = X_train_val_combined.to_numpy()
            elif isinstance(X_train_val_combined, np.ndarray):
                # If it's already a numpy array, just use it directly
                candidates = X_train_val_combined
            else:
                raise TypeError("Unsupported data type for X_train_val_combined")
        except AttributeError as e:
            print(f"AttributeError: {e}")
        except TypeError as e:
            print(f"TypeError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
        start_time_pair = time.time()
        pairwise_shap_values, pairwise_similarity_scores = pairwise_shapley_explanation(
            final_model_wrapper,#final_model, 
            dataset, explicands, candidates, candidate_indices,
            feature_names=X_train_val_combined.columns.to_list(),
             compute_sim=True)
        
    else:
        raise ValueError("Unsupported explanation method!")
    
    #########################################################
    #########################################################
    #########################################################
    #########################################################
    ## calculate shap values
    if explain_method == "Conditional-all":
        # print(explainer)
        shap_values = transform_shapr_to_shap(explainer, X_test_transformed, feature_names=X_train_val_combined.columns.to_list())
    elif explain_method in ["Pairwise-random", "Pairwise-comps", "Pairwise-sim"]:
        shap_values = pairwise_shap_values  # For compatibility
    else:
        shap_values = explainer(X_test_transformed)

    # if dataset == "hiv": #classification
    #     shap_values = shap_values[..., 0] #only visualize the contributions to first class (hence index 0)
    
    # End time measurement
    end_time = time.time()
    if explain_method in ["Pairwise-random", "Pairwise-comps", "Pairwise-sim"]:
        elapsed_time_pair = end_time - start_time_pair
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(f"Total running time (excl. comps.) of {explain_method}: {elapsed_time_pair:.2f} seconds")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    elapsed_time = end_time - start_time     
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"Total running time of {explain_method}: {elapsed_time:.2f} seconds")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


    if explain_method == "Uniform":
        if np.isscalar(shap_values.base_values):
            shap_values.base_values = np.repeat(shap_values.base_values, shap_values.values.shape[0])
        
    # save the shap values to a file using pickle
    with open(os.path.join(results_dir_exp, 'shap_values.pkl'), 'wb') as f:
        pickle.dump(shap_values, f)
    print("SHAP values saved successfully.")

    # save pairwise sim scores for pairwise method. 
    try:
        with open(os.path.join(results_dir_exp, 'pairwise_similarity_scores.pkl'), 'wb') as file:
            pickle.dump(pairwise_similarity_scores, file)
        print(f"Similarity scores have been saved to {results_dir_exp}.")
    except NameError:
        pass
    
    try:
        ## plot - global feature attribution
        shap.plots.bar(shap_values, clustering=clust, clustering_cutoff=1, max_display=X_train_val_combined.shape[1], show=False)
        plt.tight_layout() 
        plt.savefig(os.path.join(results_dir_exp, 'shap_global_bar.png'), bbox_inches='tight')
        plt.close()
    except:
        print("Skipping global bar plot due to the error") # for SamplingExplainer
    
    try:
        shap.plots.beeswarm(shap_values, max_display=X_train_val_combined.shape[1], show=False)
        plt.tight_layout() 
        plt.savefig(os.path.join(results_dir_exp, 'shap_global_beeswarm.png'), bbox_inches='tight')
        plt.close()
    except:
        print("Skipping beeswarm plot due to some error")
    
    try:
        shap.plots.heatmap(shap_values, instance_order=shap_values.sum(1))
        plt.tight_layout() 
        plt.savefig(os.path.join(results_dir_exp, 'shap_global_heatmap.png'), bbox_inches='tight')
        plt.close()
    except AttributeError as e:
        print(f"An error occurred: {e}")
        print("Skipping heatmap plot due to the AttributeError") # for SamplingExplainer
    except IndexError as e:
        print(f"An error occurred: {e}")
        print("Skipping heatmap plot due to the IndexError") # for Conditional-all
    except:
        print("Skipping heatmap plot due to some error")
        
    ##### plot - selected features global
    # selected_features = ['grade', 'sqft', 'stories', 'beds', 'bath_comb', 'condition', 'view_lakewash', 'gara_sqft', 'fbsmt_grade']

    # if model_version == "8":
    # # base_model 8 (w/ days_since_sale
    #     selected_features = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 
    #                              'grade', 'fbsmt_grade', 'condition', 
    #                              'stories', 'beds', 'bath_comb', 
    #                              'garb_sqft', 'gara_sqft', 
    #                              'wfnt', 'golf', 'greenbelt', 'noise_traffic', 
    #                              'view_rainier', 'view_olympics', 'view_cascades',
    #                              'view_territorial', 'view_skyline', 'view_sound', 
    #                              'view_lakewash', 'view_lakesamm', 'view_otherwater', 
    #                              'view_other', 'days_since_sale', 
    #                              'home_age', 'reno_age', 'location_value', 
    #                              'city_BELLEVUE', 'city_FEDERAL WAY', 'city_KENT', 
    #                              'city_KING COUNTY', 'city_KIRKLAND', 'city_RENTON', 
    #                              'city_SAMMAMISH', 'city_SEATTLE', 'city_SHORELINE', 
    #                              'city_other', 'submarket_D', 'submarket_F', 
    #                              'submarket_I', 'submarket_K', 'submarket_L', 
    #                              'submarket_M', 'submarket_O', 'submarket_Q', 
    #                              'submarket_R', 'submarket_other']
    # elif model_version == "7":
    #     # base_model 7 (w/ days_since_sale
    #     selected_features = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 
    #                              'grade', 'fbsmt_grade', 'condition', 
    #                              'stories', 'beds', 'bath_full','bath_3qtr','bath_half', 
    #                              'garb_sqft', 'gara_sqft', 
    #                              'wfnt', 'golf', 'greenbelt', 'noise_traffic', 
    #                              'view_rainier', 'view_olympics', 'view_cascades',
    #                              'view_territorial', 'view_skyline', 'view_sound', 
    #                              'view_lakewash', 'view_lakesamm', 'view_otherwater', 
    #                              'view_other', 'days_since_sale',
    #                              'home_age', 'reno_age', 'location_value', 
    #                              'city_BELLEVUE', 'city_FEDERAL WAY', 'city_KENT', 
    #                              'city_KING COUNTY', 'city_KIRKLAND', 'city_RENTON', 
    #                              'city_SAMMAMISH', 'city_SEATTLE', 'city_SHORELINE', 
    #                              'city_other', 'submarket_D', 'submarket_F', 
    #                              'submarket_I', 'submarket_K', 'submarket_L', 
    #                              'submarket_M', 'submarket_O', 'submarket_Q', 
    #                              'submarket_R', 'submarket_other']
    # elif model_version == "9":
    # # base_model 9 (w/ days_since_sale
    #     selected_features = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 'grade', 'fbsmt_grade', 
    #                              'condition', 'stories', 'beds', 'bath_comb', 'garb_sqft', 'gara_sqft', 
    #                              'wfnt', 'noise_traffic', 'view_olympics', 
    #                              'view_cascades', 'view_territorial', 'view_skyline', 'view_sound', 'view_lakewash', 
    #                              'view_lakesamm', 'days_since_sale',
    #                              'home_age', 'reno_age', 'location_value', 'city_BELLEVUE', 'city_FEDERAL WAY', 
    #                              'city_KENT', 'city_KING COUNTY', 'city_KIRKLAND', 'city_RENTON', 'city_SAMMAMISH', 
    #                              'city_SEATTLE', 'city_SHORELINE', 'city_other', 'submarket_D', 'submarket_F', 'submarket_I', 
    #                              'submarket_K', 'submarket_L', 'submarket_M', 'submarket_O', 'submarket_Q', 'submarket_R', 'submarket_other']
    # elif model_version == "10":
    #     # base_model 10 (but no days_since_sale
    #     selected_features = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt', 'grade',
    #                              'condition', 'stories', 'beds', 'bath_comb',
    #                              'wfnt', 'noise_traffic', 'view_olympics', 
    #                              'view_sound', 'view_lakewash', 
    #                              'view_lakesamm', 
    #                              'home_age', 'reno_age', 'location_value', 'city_BELLEVUE', 
    #                              'city_KIRKLAND', 'city_SEATTLE', 'city_other', 'submarket_D', 'submarket_I', 
    #                              'submarket_M', 'submarket_O', 'submarket_Q', 'submarket_R', 'submarket_other']
        
    # else:
    #     raise ValueError(f"Unexpected model_version: {model_version}")
    ## selected_features = ['grade', 'sqft', 'stories', 'beds', 'bath_full', 'condition', 'view_lakewash', 'gara_sqft', 'fbsmt_grade']
    # for feature_name in selected_features:
    #     shap.plots.scatter(shap_values[:,feature_name], ylabel="SHAP value\n(higher means higher predicted sale price)", show=False)
    #     plt.tight_layout() 
    #     plt.savefig(os.path.join(results_dir_exp, f'shap_global_feature_{feature_name}.png'), bbox_inches='tight')
    #     plt.close()

    #     # including the most interacting feature
    #     shap.plots.scatter(shap_values[:, feature_name], color=shap_values,
    #                       ylabel="SHAP value\n(higher means higher predicted sale price)", show=False)
    #     plt.tight_layout() 
    #     plt.savefig(os.path.join(results_dir_exp, f'shap_global_feature_{feature_name}_interaction.png'), bbox_inches='tight')
    #     plt.close()

    # # # some fixed features interaction
    # shap.plots.scatter(shap_values[:, 'sqft'], color=shap_values[:, "grade"],
    #                       ylabel="SHAP value\n(higher means higher predicted sale price)", show=False)
    # plt.tight_layout() 
    # plt.savefig(os.path.join(results_dir_exp, f'shap_global_feature_sqft_grade_interaction.png'), bbox_inches='tight')
    # plt.close()

    # ## plot - local feature attribution
    # random_indices = [0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203, 300, 400, 500]
    # # random_indices = np.random.choice(X_test_transformed.index, size=5, replace=False)
    # for idx in random_indices:
    #     try:
    #         plt.figure(figsize=(12, 8)) 
    #         shap.plots.waterfall(shap_values[idx], show=False)
    #         plt.tight_layout() 
    #         plt.savefig(os.path.join(results_dir_exp, f'shap_local_test_{idx}.png'), bbox_inches='tight')
    #         plt.close()
    #     except Exception as e:
    #         print(f"Failed to process index {idx}: {e}")
    #         print(f"Index: {idx}, SHAP values shape: {shap_values[idx].shape}")
    #         print(f"Index: {idx}, SHAP values: {shap_values[idx]}")

def explain_perturb(results_dir, dataset, explain_method):
    """
    Only support King County Dataset
    """
    if dataset != "kingcounty":
        raise ValueError("explain_perturb experiments currently only supports KingCounty Dataset! (perturbation for new datasets needs to be defined specifically)")
    # Create results_dir if it does not exist
    results_dir_exp = os.path.join(os.path.join(results_dir, 'explain_perturb'), explain_method)
    os.makedirs(results_dir_exp, exist_ok=True)
    # setup logging
    setup_logging(results_dir = results_dir_exp, filename = 'explain_perturb.log')
    # setup warning filter
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # set global random seed
    set_global_random_seed(42)

    # read the saved data
    X_train_val_combined = pd.read_csv(os.path.join(results_dir, 'X_train_val_combined.csv'))
    X_test = pd.read_csv(os.path.join(results_dir, 'X_test.csv'))
    y_test_pred = np.load(os.path.join(results_dir, 'y_test_pred.npy'))

    y_train_pred = pd.Series(np.load(os.path.join(results_dir, 'y_train_pred.npy')))
    y_val_pred = pd.Series(np.load(os.path.join(results_dir, 'y_val_pred.npy')))
    y_tr_val_pred_combined = pd.concat([y_train_pred, y_val_pred])
    
    # load the trained TPOT pipeline
    pipeline = joblib.load(os.path.join(results_dir, 'tpot_pipeline.pkl'))

    # transform the data if the pipeline has preprocessing steps
    X_train_val_transformed = preprocess_data(pipeline, X_train_val_combined)
    X_test_transformed = preprocess_data(pipeline, X_test)

    
    # get the final model from the pipeline
    final_model = get_final_model(pipeline)

    # clust = shap.utils.hclust(X_test, y_test_pred, linkage="single")

    # Let's get 10 random homes from the test set
    random_indices = np.random.choice(X_test.shape[0], size=100, replace=False)
    # perturbate on sqft features
    perturbations = [-500, -250, -100, -50, +50, +100, +250, +500]
    
    results_list = []

    for idx in random_indices:
        original_home = X_test_transformed.iloc[[idx]]
        original_prediction = final_model.predict(original_home)[0]
        
        # Create perturbed data by changing the 'sqft' feature
        perturbed_homes = []
        for p in perturbations:
            perturbed_home = original_home.copy()
            perturbed_home['sqft'] += p
            perturbed_homes.append(perturbed_home)
        
        perturbed_homes = pd.concat(perturbed_homes)
        perturbed_predictions = final_model.predict(perturbed_homes)
        
        # Using Shapley methods to explain the original home and the perturbed homes
        # explainer = shap.KernelExplainer(final_model.predict, X_train_val_combined.sample(n=500, random_state=42))
        if explain_method == "Baseline-median":
            explainer = shap.KernelExplainer(final_model.predict,
                                             X_train_val_combined.median().values.reshape((1, X_train_val_combined.shape[1])),
                                             feature_names = X_train_val_combined.columns.to_list())
            print(f"Using KernelExplainer [Baseline-median] for model: {final_model.__class__.__name__}")
        elif explain_method == "Baseline-0":
            explainer = shap.KernelExplainer(final_model.predict,
                                             np.zeros((1, X_train_val_combined.shape[1])),
                                             feature_names = X_train_val_combined.columns.to_list())
            print(f"Using KernelExplainer [Baseline-0] for model: {final_model.__class__.__name__}")
        elif explain_method == "Uniform": #TODO: is it uniform or marginal???
            explainer = shap.SamplingExplainer(final_model.predict,
                                             X_train_val_combined.sample(n=100, random_state=42),
                                             # X_train_val_combined.sample(n=100, random_state=42),
                                             feature_names = X_train_val_combined.columns.to_list())
            print(f"Using SamplingExplainer [Uniform] for model: {final_model.__class__.__name__}")
        elif explain_method == "Marginal-all": #TODO: to be verified!
            explainer = shap.KernelExplainer(final_model.predict,
                                             X_train_val_combined.sample(n=100, random_state=42), #change from 500 to 100 on v7 and v9
                                             feature_names = X_train_val_combined.columns.to_list())
            print(f"Using KernelExplainer [Marginal-All] for model: {final_model.__class__.__name__}")
        elif explain_method == "Marginal-kmeans": #TODO: to be verified!
            # rather than use the whole training set to estimate expected values, we summarize with
            # a set of weighted kmeans, each weighted by the number of points they represent.
            X_train_val_combined_summary = shap.kmeans(X_train_val_combined, 10)
            explainer = shap.KernelExplainer(final_model.predict,
                                             X_train_val_combined_summary,
                                             feature_names = X_train_val_combined.columns.to_list())
            print(f"Using KernelExplainer [Marginal-Kmeans10] for model: {final_model.__class__.__name__}")
        elif explain_method == "Conditional-all":     
            # use implementation of shapr: https://github.com/NorskRegnesentral/shapr.git
            # Explain the model using the empirical approach
            explainer_ori = shaprpy.explain(
                model=final_model,
                x_train=X_train_val_combined.sample(n=100, random_state=42),
                x_explain=original_home,
                approach='empirical',
                prediction_zero=y_tr_val_pred_combined.mean().item(),
                # n_combinations = 100,
                n_combinations = 100,
                n_batches = 10
            )
            explainer_pert = shaprpy.explain(
                model=final_model,
                x_train=X_train_val_combined.sample(n=100, random_state=42),
                x_explain=perturbed_homes,
                approach='empirical',
                prediction_zero=y_tr_val_pred_combined.mean().item(),
                # n_combinations = 100,
                n_combinations = 100,
                n_batches = 10
            )
            print(f"Using shaprpy-Conditional-shap for model: {final_model.__class__.__name__}")
        elif explain_method == "TreeShap-treepath":
            if is_tree_based(final_model):
                explainer = shap.TreeExplainer(final_model, feature_perturbation='tree_path_dependent')
                print(f"Using TreeExplainer for model: {final_model.__class__.__name__}")
            else:
                raise TypeError("The final_model from TPOT is not Tree-based model, thus cannot be explained by TreeSHAP!")
        
        elif explain_method == "TreeShap-interventional":
            if is_tree_based(final_model):
                explainer = shap.TreeExplainer(final_model, data = X_train_val_combined.sample(n=100, random_state=42), feature_perturbation='interventional')
                print(f"Using TreeExplainer for model: {final_model.__class__.__name__}")
            else:
                raise TypeError("The final_model from TPOT is not Tree-based model, thus cannot be explained by TreeSHAP!")
        
        # for pairwise: use the original home as the background!
        elif explain_method == "Pairwise":  
            explicands = perturbed_homes.to_numpy()
            candidates = original_home.to_numpy()
            candidate_indices = [[0]]*len(perturbations)

            pairwise_shap_values_pert, _ = pairwise_shapley_explanation(
                final_model, dataset, explicands, candidates, candidate_indices,
                feature_names=X_train_val_combined.columns.to_list(),
                compute_sim=False)

            pairwise_shap_values_ori, _ = pairwise_shapley_explanation(
                    final_model, dataset, candidates, candidates, candidate_indices,
                    feature_names=X_train_val_combined.columns.to_list(),
                    compute_sim=False)
            
        
        # # Explain the original home
        # shap_values_original = explainer.shap_values(original_home)
        # # Explain the perturbed homes
        # shap_values_perturbed = explainer.shap_values(perturbed_homes)
        ## calculate shap values
        if explain_method == "Conditional-all":
            # print(explainer)
            shap_values_ori = transform_shapr_to_shap(explainer_ori, original_home, feature_names=X_train_val_combined.columns.to_list())
            shap_values_pert = transform_shapr_to_shap(explainer_pert, perturbed_homes, feature_names=X_train_val_combined.columns.to_list())
        elif explain_method == "Pairwise":
            shap_values_ori = pairwise_shap_values_ori  # For compatibility
            shap_values_pert = pairwise_shap_values_pert
        else:
            shap_values_ori = explainer(original_home)
            shap_values_pert = explainer(perturbed_homes)

        if explain_method == "Uniform":
            if np.isscalar(shap_values_pert.base_values):
                shap_values_pert.base_values = np.repeat(shap_values_pert.base_values, shap_values_pert.values.shape[0])
            if np.isscalar(shap_values_ori.base_values):
                shap_values_ori.base_values = np.repeat(shap_values_ori.base_values, shap_values_ori.values.shape[0])

        # global plot
        
        # local plot
        # print(shap_values_ori[0]) # all zeros!
        if explain_method != "Pairwise":
            plt.figure(figsize=(12, 8)) 
            shap.plots.waterfall(shap_values_ori[0], show=False)
            plt.tight_layout() 
            plt.savefig(os.path.join(results_dir_exp, f'shap_local_test_{idx}_ori.png'), bbox_inches='tight')
            plt.close()
        
        # add credit allocation test
        features_to_include = ['grade', 'condition', 'beds', 'sqft_fbsmt', 'stories', 'sqft_1', 'sqft_lot'] #'gara_sqft'
    
        
        shap_values_ori_dict = {feature: shap_values_ori[0][feature].values for feature in features_to_include}
        shap_values_pert_dict = {feature: [shap_values_pert[i][feature].values for i in range(len(perturbations))] for feature in features_to_include}
      
        # original data
        result_dict = {
                'home_index': idx,
                'sqft_perturbation': 0,
                'predicted_price': original_prediction,
                'predicted_price_change': 0,
                'shap_value_sqft': shap_values_ori[0]['sqft'].values, # index 0 is for sqft [always] -->TODO: use key-based index?
                'shap_value_sqft_change': 0
            }
        
        for feature in features_to_include:
            result_dict[f'shap_value_{feature}'] = shap_values_ori_dict[feature]
            result_dict[f'shap_value_{feature}_change'] = 0
        
        all_shap_values_ori = shap_values_ori[0].values.flatten()
        other_shap_value_ori = sum(all_shap_values_ori) - shap_values_ori[0]['sqft'].values - sum(shap_values_ori_dict.values())
        result_dict['shap_value_others'] = other_shap_value_ori
        result_dict['shap_value_others_change'] = 0

        
        results_list.append(result_dict)
        
        for i, p in enumerate(perturbations):
            change_in_sqft = p
            change_in_price = perturbed_predictions[i] - original_prediction
            
            # Print and analyze the change in SHAP values for 'sqft' and other features
            print(f"Home idx: {idx}, sqft: {original_home[['sqft']]}, sqft perturbation: {change_in_sqft}, change in predicted price: {change_in_price}")
            print(f"Original shap value for 'sqft': {shap_values_ori[0]['sqft'].values}")
            print(f"Perturbed shap value for 'sqft': {shap_values_pert[i]['sqft'].values}")
            print(f"Change in shap value for 'sqft': {shap_values_pert[i]['sqft'].values - shap_values_ori[0]['sqft'].values}")

            result_dict = {
                'home_index': idx,
                'sqft_perturbation': change_in_sqft,
                'predicted_price': perturbed_predictions[i],
                'predicted_price_change': change_in_price,
                'shap_value_sqft': shap_values_pert[i]['sqft'].values,
                'shap_value_sqft_change': shap_values_pert[i]['sqft'].values - shap_values_ori[0]['sqft'].values
            }
            for feature in features_to_include:
                result_dict[f'shap_value_{feature}'] = shap_values_pert_dict[feature][i]
                result_dict[f'shap_value_{feature}_change'] = shap_values_pert_dict[feature][i] - shap_values_ori_dict[feature]

            all_shap_values_pert = shap_values_pert[i].values.flatten()
            other_shap_value_pert = sum(all_shap_values_pert) - shap_values_pert[i]['sqft'].values - sum(v[i] for k,v in shap_values_pert_dict.items())
            result_dict['shap_value_others'] = other_shap_value_pert
            result_dict['shap_value_others_change'] = other_shap_value_pert - other_shap_value_ori

            results_list.append(result_dict)
            
            # # Save the result to file
            # with open(os.path.join(results_dir_exp, f'shap_values_idx_{idx}_perturb_{p}.pkl'), 'wb') as f:
            #     pickle.dump((shap_values_original, shap_values_pert[i]), f)
            
            # try:
            #     plt.figure(figsize=(12, 8)) 
            #     shap.plots.waterfall(shap_values_pert[i], show=False)
            #     plt.tight_layout() 
            #     plt.savefig(os.path.join(results_dir_exp, f'shap_local_test_{idx}_perturb_{p}.png'), bbox_inches='tight')
            #     plt.close()
            # except ValueError as e:
            #     print(f"ValueError: {e}. Probably due to all zeros in shapley value / predicted price change!")
                
    
    # Save the collected results to a CSV file
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(results_dir_exp, 'perturbation_results.csv'), index=False)

    # Plot all homes on the same plot
    plt.figure(figsize=(12, 8))
    color_map = cm.get_cmap('tab10', len(random_indices)) 
    for i, idx in enumerate(random_indices):
        color = color_map(i)
        plot_data = results_df[results_df['home_index'] == idx]
        plot_data = plot_data.sort_values(by='sqft_perturbation') 
        # Plot predicted price
        plt.plot(plot_data['sqft_perturbation'], plot_data['predicted_price_change'], 
                 marker='o', linestyle='-', color=color, label=f'Home {idx} - Predicted Price Change')
        # Plot SHAP value for sqft
        plt.plot(plot_data['sqft_perturbation'], plot_data['shap_value_sqft_change'], 
                 marker='x', linestyle='--', color=color, label=f'Home {idx} - Shapley Value Change')

    plt.xlabel('Square Footage Perturbation')
    plt.ylabel('Value Change')
    plt.title('Impact of sqft Perturbation on Predicted Price and SHAP Value')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_exp, 'sqft_impact_all_homes.png'), bbox_inches='tight')
    plt.close()

    # plot aggregate (average)
    # Calculate average and standard deviation for 'predicted_price_change' and 'shap_value_sqft_change'
    avg_plot_data = results_df.groupby('sqft_perturbation').agg({
        'predicted_price_change': ['mean', 'std'],
        'shap_value_sqft_change': ['mean', 'std']
    }).reset_index()
    
    sqft_perturbations = avg_plot_data['sqft_perturbation']
    
    mean_pred_price_change = avg_plot_data['predicted_price_change']['mean']
    std_pred_price_change = avg_plot_data['predicted_price_change']['std']
    
    mean_shap_value_change = avg_plot_data['shap_value_sqft_change']['mean']
    std_shap_value_change = avg_plot_data['shap_value_sqft_change']['std']
    
    plt.figure(figsize=(12, 8))
    
    # Plot average predicted price change with standard deviation error bars
    plt.errorbar(
        sqft_perturbations, 
        mean_pred_price_change, 
        yerr=std_pred_price_change, 
        fmt='o', 
        color='black', 
        capsize=5, 
        label='Average Predicted Price Change'
    )
    
    # Plot average shap value change with standard deviation error bars
    # First plot the central values without error bars
    plt.plot(
        sqft_perturbations, 
        mean_shap_value_change, 
        'x', 
        markersize=10,
        color='red', 
        label='Average Shapley Value Change'
    )
    
    # Then plot the error bars separately as dashed lines with caps
    cap_width = 25  # Width of the caps
    for i in range(len(sqft_perturbations)):
        # Vertical dashed line
        plt.plot(
            [sqft_perturbations[i], sqft_perturbations[i]], 
            [mean_shap_value_change[i] - std_shap_value_change[i], mean_shap_value_change[i] + std_shap_value_change[i]], 
            color='red', 
            linestyle='--', 
            linewidth=2
        )
        # Top cap
        plt.plot(
            [sqft_perturbations[i] - cap_width, sqft_perturbations[i] + cap_width], 
            [mean_shap_value_change[i] - std_shap_value_change[i], mean_shap_value_change[i] - std_shap_value_change[i]], 
            color='red', 
            linestyle=':', 
            linewidth=2
        )
        # Bottom cap
        plt.plot(
            [sqft_perturbations[i] - cap_width, sqft_perturbations[i] + cap_width], 
            [mean_shap_value_change[i] + std_shap_value_change[i], mean_shap_value_change[i] + std_shap_value_change[i]], 
            color='red', 
            linestyle=':', 
            linewidth=2
        )
    
    plt.xlabel('Square Footage Perturbation')
    plt.ylabel('Value Change')
    plt.title('Impact of sqft Perturbation on Predicted Price and SHAP Value')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_exp, 'sqft_impact_all_homes_agg.png'), bbox_inches='tight')
    plt.close()

    # plot violin
    # Melt the dataframe for easier plotting with seaborn
    melted_df = results_df.melt(id_vars=["sqft_perturbation"], 
                                value_vars=["predicted_price_change", "shap_value_sqft_change"],
                                var_name="Metric", 
                                value_name="Value")
    plt.figure(figsize=(12, 8))
    # Create a violin plot
    sns.violinplot(
        x="sqft_perturbation", 
        y="Value", 
        hue="Metric", 
        data=melted_df, 
        split=True, 
        inner="quart", 
        palette={"predicted_price_change": "skyblue", "shap_value_sqft_change": "lightgreen"}
    )
    # Adjust legends and labels
    plt.xlabel('sqft Perturbation')
    plt.ylabel('Value Change ($)')
    plt.ylim(-1e6, 1.2e6)
    plt.title('Distribution of Predicted Price Change and Shapley Value Change')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_exp, 'sqft_impact_all_homes_agg_violin.png'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    model_version = args.model_version
    explain_method = args.explain_method
    n_pairs = args.num_pairs
    dataset=args.dataset
    
    print(f"##### n_pairs = {n_pairs} ########")
    results_dir = f"../results/{dataset}/base_model_{model_version}"


    # if dataset == "kingcounty":
    #     if explain_method == 'Pairwise':
    #         explain_perturb(results_dir=results_dir, dataset=dataset, explain_method=args.explain_method)
    #     elif explain_method in ["Pairwise-random", "Pairwise-comps", "Pairwise-sim"]:
    #         explain(results_dir=results_dir, dataset=dataset, explain_method=args.explain_method, model_version=model_version, n_pairs=n_pairs)
    #         explain_single(results_dir=results_dir,  dataset=dataset, explain_method=args.explain_method, model_version=model_version, n_pairs=n_pairs)
    #     else:
    #         explain(results_dir=results_dir, dataset=dataset, explain_method=args.explain_method, model_version=model_version, n_pairs=n_pairs)
    #         explain_single(results_dir=results_dir,  dataset=dataset, explain_method=args.explain_method, model_version=model_version, n_pairs=n_pairs)
    #         explain_perturb(results_dir=results_dir, dataset=dataset, explain_method=args.explain_method)
    # else:
    #     explain(results_dir=results_dir, dataset=dataset, explain_method=args.explain_method, model_version=model_version, n_pairs=n_pairs)
    #     explain_single(results_dir=results_dir,  dataset=dataset, explain_method=args.explain_method, model_version=model_version, n_pairs=n_pairs)

    # Running time compare only
    running_time = []
    for single_id in range(50):  # Run explain_single 50 times with different test samples
        elapsed_time = explain_single(results_dir=results_dir, dataset=dataset,
                                      explain_method=explain_method, model_version=model_version,
                                      n_pairs=n_pairs, single_id=single_id)
        running_time.append(elapsed_time)
        print(f"Completed instance {single_id + 1}/50: {elapsed_time:.2f} seconds")

    # Compute mean and std of execution times
    mean_time = np.mean(running_time)
    std_time = np.std(running_time)
    print("===================================")
    print(f"Mean runtime of {explain_method} over 50 runs: {mean_time:.2f} ± {std_time:.2f} seconds")
    print("===================================")
        
    
    
        
        
        



















