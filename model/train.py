import pandas as pd
import numpy as np
from tpot import TPOTRegressor, TPOTClassifier
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from matplotlib import pyplot as plt
import matplotlib as mpl
import shap
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
import joblib
import os
import random
from utils import setup_logging, set_global_random_seed, evaluate_performance_regression, feature_engineering_SMILES, feature_engineering_kingcounty, visualize_pred_regression, evaluate_performance_classification, visualize_pred_classification
import argparse
import warnings
import json


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Train and evaluate model for King County home price prediction.')
    parser.add_argument('--model_version', type=str, default='v0', 
                        choices=['v3', 'v2', 'v1', 'v0'
                                ], 
                        help='base model version to use (King County dataset only); \
                        v0 - original features; \
                        v1 - bath-combine; \
                        v2 - remove some features based on shapley value; \
                        v3 - further remove to 30 features in total')
    
    parser.add_argument('--dataset', type=str, default='kingcounty', 
                        choices=['kingcounty', 'hiv', 'polymer'
                                ], 
                        help='Dataset to use')
    return parser.parse_args()


def train_evaluate(results_dir, version, dataset):
    
    # Create results_dir if it does not exist
    os.makedirs(results_dir, exist_ok=True)
    # setup logging
    setup_logging(results_dir = results_dir, filename = 'train.log')
    # setup warning filter
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # set global random seed
    set_global_random_seed(42)
    
    ###### Step 1: Data Preprocessing ######
    if dataset == 'kingcounty':
        # Read the data
        df = pd.read_csv("../data/kaggle/kingcountysales/kingco_sales.csv", index_col=0)
        df = feature_engineering_kingcounty(df)
        # here we only have numerical features, so no feature encoding of cat features is needed for now
        if version == 'v0':
            features = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt','grade', 'fbsmt_grade', 'condition', 
                    'stories', 'beds', 'bath_full','bath_3qtr','bath_half',
                    'garb_sqft', 'gara_sqft','wfnt', 'golf', 'greenbelt', 'noise_traffic',
                    'view_rainier', 'view_olympics', 'view_cascades', 'view_territorial',
                    'view_skyline', 'view_sound', 'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other',
                    'days_since_sale', 'home_age', 'reno_age', 'location_value'
                       ]
        else:
            features = ['sqft', 'sqft_1', 'sqft_lot', 'sqft_fbsmt','grade', 'fbsmt_grade', 'condition', 
                    'stories', 'beds', 'bath_comb','garb_sqft', 'gara_sqft','wfnt', 'golf', 'greenbelt', 'noise_traffic',
                    'view_rainier', 'view_olympics', 'view_cascades', 'view_territorial',
                    'view_skyline', 'view_sound', 'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other',
                    'days_since_sale', 'home_age', 'reno_age', 'location_value'
                       ]
        # Add the new one-hot encoded categorical feature names (city and submarket)
        features.extend([col for col in df.columns if col.startswith('city_') or col.startswith('submarket_')])
        
        print("SELECTED FEATURES:", features)
        target = 'sale_price'
    
        sale_date_split = {'oldest':'2020-01-01', 'val':'2023-11-01', 'test':'2023-12-01'} 
        # Split the data by sale_date
        df_train = df.query(f"sale_date < '{sale_date_split['val']}' and sale_date >= '{sale_date_split['oldest']}'").copy()
        df_val = df.query(f"sale_date < '{sale_date_split['test']}' and sale_date >= '{sale_date_split['val']}'").copy()
        df_test = df.query(f"sale_date >= '{sale_date_split['test']}'").copy()
    
        max_date_tr = df_train['sale_date'].max() # the closest date
        min_date_tr = df_train['sale_date'].min()
        print('Min, Max date (predicted date) of training data', [min_date_tr, max_date_tr])
        df_train['max_date'] = max_date_tr
        df_train[['sale_date','max_date']] = df_train[['sale_date','max_date']].apply(pd.to_datetime) #if conversion required
        df_train['days_since_sale'] = (df_train['max_date'] - df_train['sale_date']).dt.days
        df_val['days_since_sale'] = 0
        df_test['days_since_sale'] = 0

        if version == 'v2':
            # base_model_9_UPDATE
            features_remove = ['golf', 'greenbelt', 'view_rainier', 'view_otherwater', 'view_other']
            print("Remove features that are not important based on shapley attribution: ", features_remove)
            # Remove items in features_remove from features
            features = [feature for feature in features if feature not in features_remove]
        if version == 'v3':
            features_remove = ['golf', 'greenbelt', 'view_rainier', 'view_otherwater', 'view_other',
                              'view_cascades', 'view_skyline', 'view_territorial',
                               'garb_sqft', 'fbsmt_grade', 'gara_sqft']
            print("Remove features that are not important based on shapley attribution: ", features_remove)
            # Remove items in features_remove from features
            features = [feature for feature in features if feature not in features_remove]
    
        X_train = df_train[features]
        y_train = df_train[target]
        X_val = df_val[features]
        y_val = df_val[target]
        X_test = df_test[features]
        y_test = df_test[target]
    
        def combine_one_hot_cols(df):
            # for city features
            df['city_other'] = df[['city_SAMMAMISH', 'city_KENT', 
                                             'city_FEDERAL WAY', 'city_SHORELINE', 'city_RENTON',
                                               'city_KING COUNTY','city_other']].max(axis=1)
            df = df.drop(columns=['city_SAMMAMISH', 'city_KENT', 'city_FEDERAL WAY', 'city_SHORELINE', 
                                  'city_RENTON', 'city_KING COUNTY'])
            print("Remove/combine some less important city-related features:", ['city_SAMMAMISH', 'city_KENT', 'city_FEDERAL WAY', 'city_SHORELINE', 'city_RENTON', 'city_KING COUNTY'])
            # for submarket features
            df['submarket_other'] = df[['submarket_F', 'submarket_L','submarket_K','submarket_other']].max(axis=1)
            df = df.drop(columns=['submarket_F', 'submarket_L','submarket_K'])
            print("Remove/combine some less important submarket-related features:", ['submarket_F', 'submarket_L','submarket_K'])
            return df
            
        if version == 'v3':
            X_train = combine_one_hot_cols(X_train)
            X_val = combine_one_hot_cols(X_val)
            X_test = combine_one_hot_cols(X_test)
        # Create Predefined Split; for TPOT cv optimization process -- > only df_val will be used for model validation in cv;
        # This can prevent data leakage in the model optimization process with cv
    
        # customize sklearn cross-validation iterator by indices
        validation_fold = np.append(
            -1 * np.ones(X_train.shape[0]), 
            np.zeros(X_val.shape[0])
        )
        ps = PredefinedSplit(validation_fold) #TODO: there is only one fold in cv now!
    
        # Combine train and validation datasets for TPOT
        X_train_val_combined = pd.concat([X_train, X_val])
        y_train_val_combined = pd.concat([y_train, y_val])
        # Save X_train_val_combined and X_test for later explanation reusage
        print(f"\nSaving train_val and test data to {results_dir}...")
    
        X_train_val_combined.to_csv(os.path.join(results_dir, 'X_train_val_combined.csv'), index=False)
        X_test.to_csv(os.path.join(results_dir, 'X_test.csv'), index=False)
        y_train_val_combined.to_csv(os.path.join(results_dir, 'y_train_val_combined.csv'), index=False)
        y_test.to_csv(os.path.join(results_dir, 'y_test.csv'), index=False)
    
        # Save indices (sale_id and pinx)
        train_val_indices = pd.concat([df_train[['sale_id', 'pinx']], df_val[['sale_id', 'pinx']]]).reset_index(drop=True)
        train_val_indices.to_csv(os.path.join(results_dir, 'train_val_indices.csv'), index=False)
        df_test[['sale_id', 'pinx']].to_csv(os.path.join(results_dir, 'test_indices.csv'), index=False)


    elif dataset == 'hiv': #binary
        test_size = 0.2 #TODO (stratified)
        data_path = "../data/moleculenet/HIV.csv"
        data = pd.read_csv(data_path)
        data = data.dropna()
        data.to_csv("../data/moleculenet/HIV_dropna.csv")
        smiles = data['smiles']
        y = data['HIV_active'].to_numpy()
        X = feature_engineering_SMILES(smiles, method='MACCS')

        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            smiles_train, smiles_test = smiles.iloc[train_index], smiles.iloc[test_index]
        X_val = X_train
        y_val = y_train

        pd.DataFrame(X_train).to_csv(os.path.join(results_dir, 'X_train_val_combined.csv'), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(results_dir, 'X_test.csv'), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(results_dir, 'y_train_val_combined.csv'), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(results_dir, 'y_test.csv'), index=False)
        
        # Report value counts and percentages for binary
        def report_value_counts(y, dataset_name):
            count_0 = (y == 0).sum()
            count_1 = (y == 1).sum()
            total = len(y)
            percent_0 = (count_0 / total) * 100
            percent_1 = (count_1 / total) * 100
            
            print(f"{dataset_name} - Class 0: {count_0} ({percent_0:.2f}%)")
            print(f"{dataset_name} - Class 1: {count_1} ({percent_1:.2f}%)")
        
        print("Value counts in y_train:")
        report_value_counts(y_train, "y_train")
        
        print("\nValue counts in y_test:")
        report_value_counts(y_test, "y_test")
            
    elif dataset == 'polymer':
        test_size = 0.2 
        data_path = "../data/polymer/polymer.csv"
        data = pd.read_csv(data_path)
        data = data[data['property'] == 'eps'] #only look at one property (dielectic constant)
        smiles = data['smiles']
        y = data['value'].to_numpy()
        X = feature_engineering_SMILES(smiles, method='MACCS')
        X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles, test_size=test_size, random_state=42)
        X_val = X_train
        y_val = y_train

        pd.DataFrame(X_train).to_csv(os.path.join(results_dir, 'X_train_val_combined.csv'), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(results_dir, 'X_test.csv'), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(results_dir, 'y_train_val_combined.csv'), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(results_dir, 'y_test.csv'), index=False)
        
    else:
        raise ValueError("Unsupported dataset!")

    # Report the data shape
    print("\nData Shapes:")
    print(f'Training data shape: {X_train.shape}')
    print(f'Validation data shape: {X_val.shape}')
    print(f'Test data shape: {X_test.shape}')


    ###### Step 2: Model training ######
    # define opt search space [now only considers model, no preprocessors and feature selectors]
    with open('tpot_config_kingcounty.json', 'r') as json_file:
        tpot_config_kingcounty = json.load(json_file)
    with open('tpot_config_polymer.json', 'r') as json_file:
        tpot_config_polymer = json.load(json_file)
    with open('tpot_config_hiv.json', 'r') as json_file:
        tpot_config_hiv = json.load(json_file)
    
    if os.path.exists(os.path.join(results_dir, 'tpot_pipeline.pkl')):
        print("\nLOADING TRAINED PIPELINE ...")
        pipeline_optimizer = joblib.load(os.path.join(results_dir, 'tpot_pipeline.pkl'))
    else:   
        # Train the model using TPOT with PredefinedSplit
        print("\nTRAINING AND SAVING THE PIPELINE ...")

        # for regression task
        if dataset == "kingcounty":
            print("Training a regressor ...")
            pipeline_optimizer = TPOTRegressor(generations=3, population_size=30, verbosity=1, random_state=42, cv=ps,
                                              config_dict=tpot_config_kingcounty, template='Regressor')
            pipeline_optimizer.fit(X_train_val_combined, y_train_val_combined)
        elif dataset== "polymer":
            print("Training a regressor ...")
            pipeline_optimizer = TPOTRegressor(generations=3, population_size=30, verbosity=1, random_state=42, cv=5,
                                              config_dict=tpot_config_polymer, template='Regressor')
            pipeline_optimizer.fit(X_train, y_train)
        
        # for classification task
        else:
            print("Training a classifier ...")
            pipeline_optimizer = TPOTClassifier(generations=3, population_size=30, verbosity=1, random_state=42, cv=5,
                                              config_dict=tpot_config_hiv, template='Classifier')
            pipeline_optimizer.fit(X_train, y_train)
        
        # Save the tpot trained pipeline
        pipeline_optimizer.export(os.path.join(results_dir, 'tpot_pipeline.py'))
        joblib.dump(pipeline_optimizer.fitted_pipeline_, os.path.join(results_dir, 'tpot_pipeline.pkl'))

    ###### Step 3: Model evaluation ######
    # Use the trained model for predictions

    if dataset == "kingcounty":
        y_train_pred = pipeline_optimizer.predict(X_train)
        y_val_pred = pipeline_optimizer.predict(X_val)
        y_test_pred = pipeline_optimizer.predict(X_test)
        # save pred
        np.save(os.path.join(results_dir, 'y_test_pred.npy'), y_test_pred)
        np.save(os.path.join(results_dir, 'y_val_pred.npy'), y_val_pred)
        np.save(os.path.join(results_dir, 'y_train_pred.npy'), y_train_pred)
        
        train_performance = evaluate_performance_regression(y_train, y_train_pred)
        val_performance = evaluate_performance_regression(y_val, y_val_pred)
        test_performance = evaluate_performance_regression(y_test, y_test_pred)
        # Print evaluation metrics
        print('\nTrain Performance:\n', train_performance)
        print('Validation Performance:\n', val_performance)
        print('Test Performance:\n', test_performance)
        visualize_pred_regression(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, results_dir)
    elif dataset == "polymer":
        y_train_pred = pipeline_optimizer.predict(X_train)
        y_test_pred = pipeline_optimizer.predict(X_test)
        # save pred
        np.save(os.path.join(results_dir, 'y_test_pred.npy'), y_test_pred)
        np.save(os.path.join(results_dir, 'y_train_pred.npy'), y_train_pred)
        
        train_performance = evaluate_performance_regression(y_train, y_train_pred)
        test_performance = evaluate_performance_regression(y_test, y_test_pred)
        # Print evaluation metrics
        print('\nTrain Performance:\n', train_performance)
        print('Test Performance:\n', test_performance)
        y_val, y_val_pred = None, None
        visualize_pred_regression(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, results_dir)
    else: # hiv dataset
        y_train_pred = pipeline_optimizer.predict(X_train)
        y_test_pred = pipeline_optimizer.predict(X_test)
        y_train_pred_prob = pipeline_optimizer.predict_proba(X_train)
        y_test_pred_prob = pipeline_optimizer.predict_proba(X_test)
        # save pred
        np.save(os.path.join(results_dir, 'y_test_pred.npy'), y_test_pred)
        np.save(os.path.join(results_dir, 'y_train_pred.npy'), y_train_pred)
        np.save(os.path.join(results_dir, 'y_train_pred_prob.npy'), y_train_pred_prob)
        np.save(os.path.join(results_dir, 'y_test_pred_prob.npy'),y_test_pred_prob)
        
        train_performance = evaluate_performance_classification(y_train, y_train_pred, y_train_pred_prob)
        test_performance = evaluate_performance_classification(y_test, y_test_pred, y_test_pred_prob)
        # Print evaluation metrics
        print('\nTrain Performance:\n', train_performance)
        print('Test Performance:\n', test_performance)
        y_val, y_val_pred = None, None
        visualize_pred_classification(y_train, y_train_pred, y_train_pred_prob, y_test, y_test_pred, y_test_pred_prob, results_dir)


if __name__ == "__main__":
    args = parse_args()
    version = args.model_version
    dataset = args.dataset
    results_dir = f"../results/{dataset}/base_model_{version}"
    train_evaluate(results_dir, version, dataset)
        
    


















    