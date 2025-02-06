import pandas as pd
import numpy as np
from tpot import TPOTRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, roc_curve
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
import joblib
import os
import sys
import random
import shap

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, MACCSkeys, rdMolDescriptors

def setup_logging(results_dir: str, filename: str):
    """
    Redirect stdout and stderr to a log file in the results directory
    
    Args:
        results_dir (str): Path to the directory where the log file will be saved.
        filename (str): File name of the log file (e.g.,'run.log')
    """
    log_file = os.path.join(results_dir, filename)
    sys.stdout = open(log_file, 'w')
    sys.stderr = sys.stdout



def set_global_random_seed(seed: int):
    """
    Set the global random seed for Python, NumPy, TensorFlow, and other frameworks
    to ensure reproducibility.
    
    Parameters:
    seed (int): The seed value to set for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Ensure all possible configurations are seeded in TensorFlow (if used)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        # For TensorFlow deterministic operations
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except ImportError:
        pass
    
    # Set seed for PyTorch, if you are using it
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f'Global random seed set to {seed}')

def feature_engineering_kingcounty(df):
    """
    Perform feature engineering on the King County dataset.
    Note: not including feature: days_since_sale; which is seperate (since it based on train_test_split)
    
    Args:
        df (pd.DataFrame): The input dataframe to be transformed.
    
    Returns:
        pd.DataFrame: The transformed dataframe.
    """
    print("""    
    ############################
    # Start of Feature Engineering
    ############################
    """)
    ############################
    # Start of Feature Engineering
    ############################
    # Combine baths
    df['bath_comb'] = df['bath_full'] + 0.75 * df['bath_3qtr'] + 0.5 * df['bath_half']
    
    total_data_points = df.shape[0]

    # Handle Outliers
    # (1) sqft
    # If sqft < sqft_1, set sqft = sqft_1 + sqft_fbsmt
    sqft_condition = df['sqft'] < df['sqft_1']
    df.loc[sqft_condition, 'sqft'] = df['sqft_1'] + df['sqft_fbsmt']
    num_modified_sqft = sqft_condition.sum()
    print(f"Modified {num_modified_sqft} out of {total_data_points} data points for 'sqft' condition ({num_modified_sqft/total_data_points:.2%})")
    # If sqft < 500 or sqft > 10000, remove data
    sqft_outlier_condition = (df['sqft'] < 500) | (df['sqft'] > 10000)
    num_removed_sqft = sqft_outlier_condition.sum()
    df = df[~sqft_outlier_condition]
    print(f"Removed {num_removed_sqft} out of {total_data_points} data points for 'sqft' outliers ({num_removed_sqft/total_data_points:.2%})")
    total_data_points -= num_removed_sqft

    # (2) join_status: If "join_status" is ‘rebuilt - after’, remove data point
    join_status_outlier_condition = df['join_status'] == 'rebuilt - after'
    num_removed_join_status = join_status_outlier_condition.sum()
    df = df[~join_status_outlier_condition]
    print(f"Removed {num_removed_join_status} out of {total_data_points} data points for 'join_status' ({num_removed_join_status/total_data_points:.2%})")
    total_data_points -= num_removed_join_status

    # (3) beds and bath_comb: if "beds" or "bath_full" > 10, remove data point
    bed_bath_outlier_condition = (df['beds'] > 10) | (df['bath_comb'] > 10)
    num_removed_bed_bath = bed_bath_outlier_condition.sum()
    df = df[~bed_bath_outlier_condition]
    print(f"Removed {num_removed_bed_bath} out of {total_data_points} data points for 'beds' or 'bath_comb' ({num_removed_bed_bath/total_data_points:.2%})")
    total_data_points -= num_removed_bed_bath

    # (4) fbsmt_grade: If fbsmt_grade = 0, impute zeros with the median value of fbsmt_grade (excluding zeros)
    fbsmt_grade_median = df.loc[df['fbsmt_grade'] != 0, 'fbsmt_grade'].median()
    fbsmt_grade_outlier_condition = df['fbsmt_grade'] == 0
    num_imputed_fbsmt_grade = fbsmt_grade_outlier_condition.sum()
    df.loc[fbsmt_grade_outlier_condition, 'fbsmt_grade'] = fbsmt_grade_median
    print(f"Imputed {num_imputed_fbsmt_grade} out of {total_data_points} data points for 'fbsmt_grade' ({num_imputed_fbsmt_grade/total_data_points:.2%})")

    # Additional Features
    
    df['sale_date_year'] = pd.to_datetime(df['sale_date']).dt.year
    ## base_model_4_UPDATE_2_homeage:  home_age (year) = sale_date - yearbuilt 
    df['home_age'] = df['sale_date_year'] - df['year_built']
    ## base_model_4_UPDATE_3_renoage:  reno_age (year) = sale_date - reno_year
    df['reno_year_adjusted'] = np.where(df['year_reno'] == 0, df['year_built'], df['year_reno'])
    df['reno_age'] = df['sale_date_year'] - df['reno_year_adjusted']
    # base_model_6_UPDATE: Create location_value feature
    df['location_value'] = df['land_val'] / df['sqft_lot']

    # Process categorical features: base_model_5_UPDATE: add city and submarket
    for cat in ['city', 'submarket']:
        top_categories = df[cat].value_counts().nlargest(10).index
        df[cat] = np.where(df[cat].isin(top_categories), df[cat], 'other')

    df = pd.get_dummies(df, columns=['city', 'submarket'], drop_first=True, dtype=int)
    
    ############################
    # End of Feature Engineering
    ############################
    
    return df

def feature_engineering_SMILES(smiles_list: list[str], method: str = 'Morgan', 
                          radius: int = 2, n_bits: int = 2048, 
                          return_bit_info=False) -> np.ndarray:
    """
    Convert SMILES strings to numerical fingerprints based on the specified method using RDKit.
    Optionally returns bit information for Morgan fingerprints.
    
    Args:
        smiles_list (list[str]): List of SMILES strings.
        method (str): Method of fingerprinting. Supported methods: 'Morgan', 'RDKit', 'MACCS', 'TopologicalTorsion', 'AtomPair'.
        radius (int): Radius of the fingerprint (applicable for Morgan and TopologicalTorsion).
        n_bits (int): Length of the fingerprint (applicable for Morgan, RDKit, and AtomPair).

    Returns:
        np.ndarray: Array of numerical fingerprints.
    """
    fingerprints = []
    all_bit_info = []  # Store bit information for all molecules
    
    # ignore the molecule, just save all the bits in the dataset in one
    list_bits = []
    legends = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:  # Ensure valid molecule
            continue

        if method == 'Morgan':
            bitInfo = {}
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bitInfo)
            all_bit_info.append(bitInfo)
            for x in fingerprint.GetOnBits():
                for i in range(len(bitInfo[x])):
                    list_bits.append((mol,x,bitInfo,i))
                    legends.append(str(x))

        elif method == 'RDKit':
            bitInfo = {}
            fingerprint = Chem.RDKFingerprint(mol, maxPath=radius, fpSize=n_bits, bitInfo = bitInfo)
            bit_tuples = [(mol, k, bitInfo) for k in bitInfo.keys()]
            all_bit_info.extend(bit_tuples)
            for x in fingerprint.GetOnBits():
                legends.append(str(x))

        elif method == 'MACCS':
            fingerprint = MACCSkeys.GenMACCSKeys(mol)
        elif method == 'TopologicalTorsion':
            fingerprint = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
        elif method == 'AtomPair':
            fingerprint = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        else:
            raise ValueError(f"Unsupported fingerprint method: {method}")

        fingerprints.append(np.array(fingerprint))

    if return_bit_info:
        return np.array(fingerprints), all_bit_info, list_bits, legends
    return np.array(fingerprints)
    

def percent_error(y_true, y_pred):
    """
    Percent Error Calculation
    """
    return (y_pred - y_true) / y_true


def evaluate_performance_regression(y_true, y_pred):
    
    # Define percentage metrics
    pct_metrics = {
        'MAPE': lambda x: np.abs(x).mean(),
        'MdPE': lambda x: np.median(x),
        # 'MdPE': lambda x: x.median(),
        'MdAPE': lambda x: np.median(np.abs(x)),
        'PE05': lambda x: (np.abs(x) < 0.05).sum() / len(x),
        'PE30': lambda x: (np.abs(x) < 0.3).sum() / len(x),
        'PE50': lambda x: (np.abs(x) < 0.5).sum() / len(x),
    }

    pe = percent_error(y_true, y_pred)
    performance = {name: func(pe) for name, func in pct_metrics.items()}
    performance.update({
        'R2': r2_score(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred, squared=False)
    })

    # Round metrics results to 3 decimal places
    rounded_performance = {k: round(v, 3) for k, v in performance.items()}
    return rounded_performance

def evaluate_performance_classification(y_true, y_pred, y_pred_prob=None):
    """
    Evaluate the performance of a classification model.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    y_pred_prob (array-like): Predicted probabilities (for AUC calculation).
    
    Returns:
    dict: A dictionary containing various classification performance metrics.
    """
    
    classification_metrics = {
        'Accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred),
        'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
        'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        'Confusion Matrix': lambda y_true, y_pred: confusion_matrix(y_true, y_pred)
    }
    
    performance = {name: func(y_true, y_pred) for name, func in classification_metrics.items()}
    
    # Calculate AUC if predicted probabilities are provided
    if y_pred_prob is not None:
        performance['AUC'] = roc_auc_score(y_true, y_pred_prob[:, 1])
    
    return performance




def visualize_pred_regression(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, results_dir):
    """
    Visualize the regression model predictions.
    
    Parameters:
    y_train (array-like): Actual training values.
    y_train_pred (array-like): Predicted training values.
    y_val (array-like): Actual validation values.
    y_val_pred (array-like): Predicted validation values.
    y_test (array-like): Actual test values.
    y_test_pred (array-like): Predicted test values.
    results_dir (str): Directory to save the plots.
    """
    mpl.rcParams["figure.dpi"] = 300
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.titlesize'] = 22
    mpl.rcParams['legend.fontsize'] = 18
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['ytick.labelsize'] = 18
    
    # Plot parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_train, y_train_pred, label='Train', alpha=0.7)
    if y_val is not None:
        plt.scatter(y_val, y_val_pred, label='Validation', alpha=0.7)
    plt.scatter(y_test, y_test_pred, label='Test', alpha=0.7)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--k', lw=2)
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Parity Plot')
    plt.tight_layout()
    plt.grid(False)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'prediction_parity_plot.pdf'))
    
    # Plot parity plot -- log-scale
    plt.figure(figsize=(8, 8))
    plt.scatter(y_train, y_train_pred, label='Train', alpha=0.7)
    if y_val is not None:
        plt.scatter(y_val, y_val_pred, label='Validation', alpha=0.7)
    plt.scatter(y_test, y_test_pred, label='Test', alpha=0.7)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--k', lw=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Parity Plot (Log Scale)')
    plt.tight_layout()
    plt.grid(False)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'prediction_parity_plot_log.pdf'))


def visualize_pred_classification(y_train, y_train_pred, y_train_prob, y_test, y_test_pred, y_test_prob, results_dir):
    """
    Visualize the classification model performance.
    
    Parameters:
    y_train (array-like): Actual training labels.
    y_train_pred (array-like): Predicted training labels.
    y_train_prob (array-like): Predicted probabilities of training labels.
    y_test (array-like): Actual test labels.
    y_test_pred (array-like): Predicted test labels.
    y_test_prob (array-like): Predicted probabilities of test labels.
    results_dir (str): Directory to save the plots.
    """
    mpl.rcParams["figure.dpi"] = 300
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.titlesize'] = 22
    mpl.rcParams['legend.fontsize'] = 18
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['ytick.labelsize'] = 18
    
    # Plot confusion matrix for training data
    cm_train = confusion_matrix(y_train, y_train_pred)
    cmd_train = ConfusionMatrixDisplay(cm_train)
    plt.figure(figsize=(8, 8))
    cmd_train.plot(ax=plt.gca(), cmap='Blues')
    plt.title('Confusion Matrix - Train')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_train.pdf'))
    
    # Plot confusion matrix for test data
    cm_test = confusion_matrix(y_test, y_test_pred)
    cmd_test = ConfusionMatrixDisplay(cm_test)
    plt.figure(figsize=(8, 8))
    cmd_test.plot(ax=plt.gca(), cmap='Blues')
    plt.title('Confusion Matrix - Test')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_test.pdf'))

    # Plot ROC curve for training and test data
    plt.figure(figsize=(8, 8))
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob[:, 1])
    roc_auc_train = roc_auc_score(y_train, y_train_prob[:, 1])
    plt.plot(fpr_train, tpr_train, label=f'Train (AUC = {roc_auc_train:.2f})')

    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob[:, 1])
    roc_auc_test = roc_auc_score(y_test, y_test_prob[:, 1])
    plt.plot(fpr_test, tpr_test, label=f'Test (AUC = {roc_auc_test:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curve.pdf'))

# class ShapValues:
#     """
#     A class to mimic the output structure of SHAP values produced by the `shap` library.

#     Attributes
#     ----------
#     values : numpy.ndarray
#         Array containing the SHAP values for the predictions.
#     base_values : float or numpy.ndarray
#         Baseline value(s) used to compute SHAP values, typically the mean prediction value.
#     data : numpy.ndarray
#         Array containing the input features data on which SHAP values are computed.
#     """
#     def __init__(self, values, base_values, data):
#         """
#         Initializes the ShapValues object with SHAP values, baseline values, and input data.

#         Parameters
#         ----------
#         values : numpy.ndarray
#             The SHAP values for the predictions.
#         base_values : float or numpy.ndarray
#             The base value(s) associated with the model's predictions.
#         data : numpy.ndarray
#             The input data used to compute SHAP values.
#         """
#         self.values = values
#         self.base_values = base_values
#         self.data = data

def transform_shapr_to_shap(explain_output, X_test_transformed, feature_names):
    """
    Transforms the output of the `shapr` explain() function to match the data structure expected by the `shap` library.

    Parameters
    ----------
    explain_output : tuple
        The output from the `shapr` package's explain() function, which is expected to be a tuple containing:
        - Pandas DataFrame with SHAP values (first column should be base values, the rest are SHAP values).
        - Numpy Array with model predictions on `x_explain`.
        - Dictionary containing additional information.
        - Dictionary containing elapsed time information (if timing is set to True).
        - Dictionary containing MSEv evaluation criterion scores.
    X_test_transformed : pandas.DataFrame
        The transformed test data.
    feature_names : list
        A list of feature names.
    Returns
    -------
    shap.Explanation
        An object containing SHAP values, baseline values, and input data, structured similarly to `shap` library outputs.
    """
    shap_vals_df, predictions, additional_info, timing_info, mse_info = explain_output

    # Extract base values (first column) and SHAP values (remaining columns)
    base_values = shap_vals_df.iloc[:, 0].values
    shap_values = shap_vals_df.iloc[:, 1:].values
    
    # print("<<< shap_values")
    # print(shap_values)

    # print("<<< base_values")
    # print(base_values)

    # Ensure the input data has the same order and format as shap values
    data = X_test_transformed.to_numpy()
    # print("<<< data")
    # print(data)

    # Create a shap.Explanation object
    explanation = shap.Explanation(values=shap_values, base_values=base_values, data=data, feature_names=feature_names)

    return explanation


