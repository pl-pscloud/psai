import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingRegressor, StackingClassifier, VotingRegressor, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split, GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_log_error, root_mean_squared_error, mean_absolute_percentage_error, accuracy_score, f1_score, roc_auc_score, log_loss, make_scorer
from sklearn.base import clone
import pandas as pd
import numpy as np
import pickle
import os

from psai.scalersencoders import create_preprocessor
from psai.pstorch import PyTorchRegressor, RMSLELoss, PyTorchClassifier

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor

pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, 
        max_resource=5,  # Number of folds
        reduction_factor=3
    )

sampler = optuna.samplers.TPESampler(seed=42)

best_classification_cv = {
    'lightgbm': 0,
    'xgboost': 0,
    'catboost': 0,
    'pytorch': 0
}
best_classification_model = {
    'lightgbm': None,
    'xgboost': None,
    'catboost': None,
    'pytorch': None
}

best_regression_cv = {
    'lightgbm': 1e10,
    'xgboost': 1e10,
    'catboost': 1e10,
    'pytorch': 1e10
}
best_regression_model = {
    'lightgbm': None,
    'xgboost': None,
    'catboost': None,
    'pytorch': None
}

class psML:
    """
    Machine Learning Pipeline for training, evaluation, and prediction
    """
    def __init__(self, config, X, y):
        self.models = {}
        self.config = config

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['dataset']['test_size'], random_state=self.config['dataset']['random_state'])

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.columns = {}
        self.preprocessor = None
        
        
    def save_model(self, filepath):
        """
        Save the pSML object to a file using pickle
        
        Parameters:
        -----------
        filepath : str
            Path where the model will be saved
        
        Returns:
        --------
        None
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model successfully saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load a pSML object from a file
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model file
        
        Returns:
        --------
        psML
            Loaded pSML object
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model successfully loaded from {filepath}")
        return model

    def optimize_all_models(self):

        self.create_preprocessor()

        if self.config['dataset']['task_type'] == 'classification':
    
            if self.config['models']['lightgbm']['enabled']:
                self.optimize_model('lightgbm')

            if self.config['models']['xgboost']['enabled']:
                self.optimize_model('xgboost')

            if self.config['models']['catboost']['enabled']:
                self.optimize_model('catboost')
            
            if self.config['models']['pytorch']['enabled']:
                self.optimize_model('pytorch')

        if self.config['dataset']['task_type'] == 'regression':
        
            if self.config['models']['lightgbm']['enabled']:
                self.optimize_model('lightgbm')

            if self.config['models']['xgboost']['enabled']:
                self.optimize_model('xgboost')

            if self.config['models']['catboost']['enabled']:
                self.optimize_model('catboost')
        
            if self.config['models']['pytorch']['enabled']:
                self.optimize_model('pytorch')

        
    def create_preprocessor(self):
        preprocessor, columns = create_preprocessor(self.config['preprocessor'], self.X_train)
        self.preprocessor = preprocessor
        self.columns = columns

    def scores(self):
        if self.config['models']['lightgbm']['enabled']:
            print(f'LightGBM CV Score: {self.models["lightgbm"]["cv_score"]} , Test Score: {self.models["lightgbm"]["final_model"]["score"]}')

        if self.config['models']['xgboost']['enabled']:
            print(f'XGBoost CV Score: {self.models["xgboost"]["cv_score"]} , Test Score: {self.models["xgboost"]["final_model"]["score"]}')

        if self.config['models']['catboost']['enabled']:
            print(f'CatBoost CV Score: {self.models["catboost"]["cv_score"]} , Test Score: {self.models["catboost"]["final_model"]["score"]}')
        
        if self.config['models']['pytorch']['enabled']:
            print(f'PyTorch CV Score: {self.models["pytorch"]["cv_score"]} , Test Score: {self.models["pytorch"]["final_model"]["score"]}')

    def get_evaluation_metric(self,metric_name):
       
        metric_mapping = {
            'acc': accuracy_score,
            'f1': lambda y_true, y_pred: f1_score(y_true, (y_pred >= 0.5).astype(int), average='binary'),
            'auc': roc_auc_score,
            'prec': lambda y_true, y_pred: precision_score(y_true, (y_pred >= 0.5).astype(int), average='binary'),
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error,
            'mape': mean_absolute_percentage_error,

        }
        
        if metric_name not in metric_mapping:
            print(f"Warning: Metric '{metric_name}' not found. Using roc_auc_score as default.")
            return roc_auc_score
        
        return metric_mapping[metric_name]

    def optimize_model(self, model_name):
        
        self.models[model_name] = {}
        
        def objective(trial):
            global best_classification_cv
            global best_classification_model
            global best_regression_cv
            global best_regression_model


            if model_name == 'lightgbm':
                params = {
                    "objective": self.config['models']['lightgbm']['params']['objective'],
                    "device": self.config['models']['lightgbm']['params']['device'],
                    "metric": self.config['models']['lightgbm']['params']['eval_metric'], 
                    "verbosity": -1,
                    "n_estimators": 10000, #trial.suggest_int('n_estimators', 500, 3500),
                    "boosting_type": trial.suggest_categorical('boosting_type', ['gbdt','goss']), # goss, gbdt, dart, rf
                    "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
                    "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                    "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.2),
                    "min_split_gain": trial.suggest_loguniform("min_split_gain", 1e-8, 1.0),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "num_threads": self.config['models']['lightgbm']['params']['num_threads'],
                    "verbose": self.config['models']['lightgbm']['params']['verbose']
                }

                if params['boosting_type'] != 'goss':
                    params["bagging_fraction"] = trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                    params["bagging_freq"] =  trial.suggest_int("bagging_freq", 1, 7),
            
            if model_name == 'xgboost':
                params = {
                    "objective": self.config['models']['xgboost']['params']['objective'],
                    "eval_metric": self.config['models']['xgboost']['params']['eval_metric'],
                    "device": self.config['models']['xgboost']['params']['device'],
                    "booster": trial.suggest_categorical('booster', ['gbtree']),
                    "max_depth": trial.suggest_int('max_depth' ,3, 20),
                    "learning_rate": trial.suggest_loguniform('learning_rate', 0.001, 0.2),
                    "n_estimators": trial.suggest_int('n_estimators', 500, 3000),
                    "subsample": trial.suggest_float('subsample', 0, 1),
                    "lambda": trial.suggest_float('lambda', 1e-4, 5, log=True),
                    "gamma": trial.suggest_float('gamma', 1e-4, 5, log=True),
                    "alpha": trial.suggest_float('alpha', 1e-4, 5, log=True),
                    "min_child_weight": trial.suggest_categorical('min_child_weight', [0.5,1,3,5]),
                    "colsample_bytree": trial.suggest_float('colsample_bytree', 0.5, 1),
                    "colsample_bylevel": trial.suggest_float('colsample_bylevel', 0.5, 1),
                    "early_stopping_rounds": 100,
                    "nthread": self.config['models']['xgboost']['params']['nthread'],
                    "verbose": self.config['models']['xgboost']['params']['verbose']
                    
                }
                
            if model_name == 'catboost':
                params = {
                    'loss_function': self.config['models']['catboost']['params']['objective'],
                    'eval_metric': self.config['models']['catboost']['params']['eval_metric'],
                    'task_type': 'CPU' if self.config['models']['catboost']['params']['device'] == 'cpu' else 'GPU',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.2),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
                    'border_count': trial.suggest_int('border_count', 32, 128),
                    'random_seed': 42,
                    'verbose': False,
                    'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                    #'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
                    'feature_border_type': trial.suggest_categorical('feature_border_type', ['Median', 'Uniform', 'GreedyMinEntropy']),
                    'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
                    'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
                    'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
                    'thread_count': self.config['models']['catboost']['params']['thread_count'],
                    'verbose': self.config['models']['catboost']['params']['verbose']
                }
                if params['bootstrap_type'] == 'Bernoulli':
                    params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
                if params['bootstrap_type'] == 'Bayesian':
                    params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
                if params['grow_policy'] == 'Lossguide':
                    params['max_leaves'] = trial.suggest_int('max_leaves', 2, 32)
                
            if model_name == 'pytorch':
                params = {
                    "learning_rate": trial.suggest_categorical('learning_rate', self.config['models']['pytorch']['params']['learning_rate']),
                    "optimizer_name": trial.suggest_categorical('optimizer_name', self.config['models']['pytorch']['params']['optimizer_name']),
                    "batch_size": trial.suggest_categorical('batch_size', self.config['models']['pytorch']['params']['batch_size']),
                    "net":  trial.suggest_categorical('net', self.config['models']['pytorch']['params']['net']),
                    "weight_init": trial.suggest_categorical('weight_init', self.config['models']['pytorch']['params']['weight_init']),
                    "verbose": self.config['models']['pytorch']['params']['verbose'],
                    "device": 'cuda' if self.config['models']['pytorch']['params']['device'] == 'gpu' else 'cpu',
                    "max_epochs": self.config['models']['pytorch']['params']['train_max_epochs'],
                    "patience": self.config['models']['pytorch']['params']['train_patience'],
                    "loss": self.config['models']['pytorch']['params']['objective'],
                    "embedding_info": self.columns['embedding_info'],
                    "verbose": self.config['models']['pytorch']['params']['verbose']
                }
                

            if self.config['dataset']['verbose'] > 0:
                print(f'===============  {model_name} training - trail {trial.number}  =========================')
                
            
            split = self.config['dataset']['cv_folds']
            scores = 0

            if self.config['dataset']['task_type'] == 'classification':
                skf = StratifiedKFold(n_splits=split, shuffle=True, random_state=self.config['dataset']['random_state'])
            else:
                skf = KFold(n_splits=split, shuffle=True, random_state=self.config['dataset']['random_state'])
                
            fitted_trial_models = {}

            for fold, (train_index, val_index) in enumerate(skf.split(self.X_train, self.y_train), 1):
                
                if self.config['dataset']['verbose'] > 1:
                    print(f'Fold {fold} ...', end="")
                
                X_train_fold, X_val_fold = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
                y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]
                
                fold_preprocessor = clone(self.preprocessor)
                fold_preprocessor.fit(X_train_fold, y_train_fold)
                # Transform training and validation data
                X_val_transformed = fold_preprocessor.transform(X_val_fold.reset_index(drop=True))

                if self.config['dataset']['task_type'] == 'classification':
                    if model_name == 'lightgbm':
                        clf = LGBMClassifier(**params)
                    elif model_name == 'xgboost':
                        clf = XGBClassifier(**params)
                    elif model_name == 'catboost':
                        clf = CatBoostClassifier(**params)
                    elif model_name == 'pytorch':
                        clf = PyTorchClassifier(**params)
                else:
                    if model_name == 'lightgbm':
                        clf = LGBMRegressor(**params)
                    elif model_name == 'xgboost':
                        clf = XGBRegressor(**params)
                    elif model_name == 'catboost':
                        clf = CatBoostRegressor(**params)
                    elif model_name == 'pytorch':
                        clf = PyTorchRegressor(**params)

                pipeline = Pipeline([
                            ('preprocessor', fold_preprocessor),
                            (model_name, clf)  
                        ])
                        
                try:
                    if model_name == 'lightgbm':
                        pipeline.fit(
                            X_train_fold.reset_index(drop=True), 
                            y_train_fold.reset_index(drop=True),
                            lightgbm__eval_set=[(X_val_transformed, y_val_fold.reset_index(drop=True))],
                            lightgbm__eval_metric=self.config['models']['lightgbm']['params']['eval_metric'],
                            lightgbm__callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
                        )
                    if model_name == 'xgboost':
                        pipeline.fit(
                            X_train_fold.reset_index(drop=True), 
                            y_train_fold.reset_index(drop=True),
                            xgboost__eval_set=[(X_val_transformed, y_val_fold.reset_index(drop=True))],
                            xgboost__verbose=False,
                        )
                    if model_name == 'catboost':
                        pipeline.fit(
                            X_train_fold.reset_index(drop=True), 
                            y_train_fold.reset_index(drop=True),
                            catboost__eval_set=[(X_val_transformed, y_val_fold.reset_index(drop=True))],
                            catboost__early_stopping_rounds=100, 
                        )
                    if model_name == 'pytorch':
                        pipeline.fit(
                            X_train_fold, 
                            y_train_fold,
                            pytorch__eval_set = [X_val_transformed, y_val_fold]
                        )
                        
                    
                except optuna.exceptions.TrialPruned:
                    if self.config['dataset']['verbose'] > 1:
                        print(f"Trial {trial.number} pruned at fold {fold}.")
                    raise  # Re-raise to notify Optuna)
                
                if self.config['dataset']['task_type'] == 'classification':
                    y_pred = pipeline.predict_proba(X_val_fold.reset_index(drop=True))[:, 1]
                else:
                    y_pred = pipeline.predict(X_val_fold.reset_index(drop=True))
                
                s = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(y_val_fold.reset_index(drop=True), y_pred)

                df_y_pred = pd.DataFrame(y_pred)
                df_y_pred.index = y_val_fold.index
                df_y_pred.columns = [y_val_fold.columns[0]]

                fitted_trial_models[f'fold{fold}'] = {'model': pipeline, 'oof': df_y_pred}
                
                scores += s
                
                if self.config['dataset']['verbose'] > 1:
                    print(f" score: {s}")

                # Report intermediate score
                intermediate_score = np.mean(scores)
                trial.report(intermediate_score, step=fold)

                # Prune trial if necessary
                if trial.should_prune():
                    if self.config['dataset']['verbose'] > 1:
                        print(f"Trial {trial.number} pruned after fold {fold}.")
                    raise optuna.exceptions.TrialPruned()
                
            score = scores / split

            if self.config['dataset']['task_type'] == 'classification':
                if score > best_classification_cv[model_name]:
                    best_classification_cv[model_name] = score
                    best_classification_model[model_name] = fitted_trial_models
            else:
                if score < best_regression_cv[model_name]:
                    best_regression_cv[model_name] = score
                    best_regression_model[model_name] = fitted_trial_models
            
            if self.config['dataset']['verbose'] > 0:
                print(f"CV {self.config['models'][model_name]['optuna_metric']} Score:", score)

            return score

        # Create an Optuna study
        if self.config['dataset']['task_type'] == 'classification':
            study = optuna.create_study(direction='maximize', study_name=f'{model_name} Optimization', pruner=pruner, sampler=sampler)
        else:
            study = optuna.create_study(direction='minimize', study_name=f'{model_name} Optimization', pruner=pruner, sampler=sampler)
        
        # Optimize the objective function
        study.optimize(objective, n_trials=self.config['models'][model_name]['optuna_trials'], n_jobs=self.config['models'][model_name]['optuna_n_jobs'])

        # Print study statistics
        if self.config['dataset']['verbose'] > 0:
            print("Study statistics:")
            print("  Number of finished trials: ", len(study.trials))
            print("  Best trial:")
        trial = study.best_trial

        if self.config['dataset']['verbose'] > 0:
            print("    Value: {:.4f}".format(trial.value))
            print("    Params: ")
            for key, value in trial.params.items():
                print(f"      {key}: {value}")

        if self.config['dataset']['task_type'] == 'classification':
            self.models[model_name]['cv_score'] = best_classification_cv[model_name]
            self.models[model_name]['cv_model'] = best_classification_model[model_name]
        else:
            self.models[model_name]['cv_score'] = best_regression_cv[model_name]
            self.models[model_name]['cv_model'] = best_regression_model[model_name]
        
        self.models[model_name]['best_params'] = trial.params
        self.models[model_name]['study'] = study

        p = trial.params

        if model_name == 'lightgbm':
            params = {
                "objective": self.config['models']['lightgbm']['params']['objective'],
                "device": self.config['models']['lightgbm']['params']['device'],
                "metric": self.config['models']['lightgbm']['params']['eval_metric'],
                "verbosity": -1,
                "n_estimators": 10000, #p["n_estimators"],
                "boosting_type": p["boosting_type"],
                "lambda_l1": p["lambda_l1"],
                "lambda_l2": p["lambda_l2"],
                "num_leaves": p["num_leaves"],
                "feature_fraction": p["feature_fraction"],
                "min_child_samples": p["min_child_samples"],
                "learning_rate": p["learning_rate"],
                "min_split_gain": p["min_split_gain"],
                "max_depth": p["max_depth"],
                "num_threads": self.config['models']['lightgbm']['params']['num_threads'],
                "verbose": self.config['models']['lightgbm']['params']['verbose']
            }

            if p['boosting_type'] != 'goss':
                params["bagging_fraction"] = p["bagging_fraction"]
                params["bagging_freq"] =  p["bagging_freq"]

        if model_name == 'xgboost':
            params = {
                "objective": self.config['models']['xgboost']['params']['objective'],
                "eval_metric": self.config['models']['xgboost']['params']['eval_metric'],
                "device": self.config['models']['xgboost']['params']['device'],
                "booster": p['booster'],
                "max_depth": p['max_depth'],
                "learning_rate": p['learning_rate'],
                "n_estimators": p['n_estimators'],
                "subsample": p['subsample'],
                "lambda": p['lambda'],
                "gamma": p['gamma'],
                "alpha": p['alpha'],
                "min_child_weight": p['min_child_weight'],
                "colsample_bytree": p['colsample_bytree'],
                "colsample_bylevel": p['colsample_bylevel'],
                "nthread": self.config['models']['xgboost']['params']['nthread'],  # Use all available cores
                "verbose": self.config['models']['xgboost']['params']['verbose']
                            }
        
        if model_name == 'catboost':
            params = {
                'loss_function': self.config['models']['catboost']['params']['objective'],
                'eval_metric': self.config['models']['catboost']['params']['eval_metric'],
                'task_type': 'CPU' if self.config['models']['catboost']['params']['device'] == 'cpu' else 'GPU',
                'n_estimators': p['n_estimators'],
                'learning_rate': p['learning_rate'],
                'depth': p['depth'],
                'l2_leaf_reg': p['l2_leaf_reg'],
                'border_count': p['border_count'],
                'random_seed': 42,
                'verbose': self.config['models']['catboost']['params']['verbose'],
                'bootstrap_type': p['bootstrap_type'],
                #'boosting_type': p['boosting_type'],
                'feature_border_type': p['feature_border_type'],
                'leaf_estimation_iterations': p['leaf_estimation_iterations'],
                'min_data_in_leaf': p['min_data_in_leaf'],
                'random_strength': p['random_strength'],
                'grow_policy': p['grow_policy'],
                'thread_count': self.config['models']['catboost']['params']['thread_count'],  # Use all available cores
            }
            if p['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = p['subsample']
            if p['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = p['bagging_temperature']
            if p['grow_policy'] == 'Lossguide':
                    params['max_leaves'] = p['max_leaves']

        if model_name == 'pytorch':
            params = {
                "learning_rate": p["learning_rate"],
                "optimizer_name": p["optimizer_name"],
                "batch_size": p["batch_size"],
                "weight_init": p["weight_init"],
                "max_epochs": self.config['models']['pytorch']['params']['final_max_epochs'],
                "patience":  self.config['models']['pytorch']['params']['final_patience'],
                "net": p['net'],
                "verbose":  self.config['models']['pytorch']['params']['verbose'],
                "device": 'cuda' if self.config['models']['pytorch']['params']['device'] == 'gpu' else 'cpu',
                "loss":  self.config['models']['pytorch']['params']['objective'],
                "embedding_info": self.columns['embedding_info'],
                "num_threads": self.config['models']['pytorch']['params']['num_threads'],  # Use all available cores for data loading
                "verbose": self.config['models']['pytorch']['params']['verbose']
            }
        
        final_preprocessor = clone(self.preprocessor)
        final_preprocessor.fit(self.X_train, self.y_train)
        # Transform training and validation data
        X_test_transformed = final_preprocessor.transform(self.X_test)

        if self.config['dataset']['task_type'] == 'classification':
            if model_name == 'lightgbm':
                clf2 = LGBMClassifier(**params)
            elif model_name == 'xgboost':
                clf2 = XGBClassifier(**params)
            elif model_name == 'catboost':
                clf2 = CatBoostClassifier(**params)
            elif model_name == 'pytorch':
                clf2 = PyTorchClassifier(**params)
        elif self.config['dataset']['task_type'] == 'regression':
            if model_name == 'lightgbm':
                clf2 = LGBMRegressor(**params)
            elif model_name == 'xgboost':
                clf2 = XGBRegressor(**params)
            elif model_name == 'catboost':
                clf2 = CatBoostRegressor(**params)
            elif model_name == 'pytorch':
                clf2 = PyTorchRegressor(**params)

        final_pipeline = Pipeline([
            ('preprocessor', final_preprocessor),
            (model_name, clf2)
        ])

        

        if model_name == 'lightgbm':
            print(f'===============  LightGBM Final Training  ============================')
            final_pipeline.fit(
                self.X_train, 
                self.y_train,
                lightgbm__eval_set=[(X_test_transformed, self.y_test)],
                lightgbm__eval_metric=self.config['models']['lightgbm']['params']['eval_metric'],
                lightgbm__callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )
        if model_name == 'xgboost':
            print(f'===============  XGBoost Final Training  ============================')
            final_pipeline.fit(
                self.X_train, 
                self.y_train,
                xgboost__eval_set=[(X_test_transformed, self.y_test)],
                xgboost__verbose=False
            )
        if model_name == 'catboost':
            print(f'===============  CatBoost Final Training  ============================')
            final_pipeline.fit(
                self.X_train, 
                self.y_train,
                catboost__eval_set=[(X_test_transformed, self.y_test)],
                catboost__early_stopping_rounds=100, 
            )
        if model_name == 'pytorch':
            print(f'===============  PyTorch Final Training  ============================')
            final_pipeline.fit(
                self.X_train, 
                self.y_train,
                pytorch__eval_set=[X_test_transformed, self.y_test],
            )
        
        if self.config['dataset']['task_type'] == 'classification':
            final_ypred = final_pipeline.predict_proba(self.X_test)
            final_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, final_ypred[:, 1])
        else:
            final_ypred = final_pipeline.predict(self.X_test)
            final_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, final_ypred)

        self.models[model_name]['final_model'] = {
            'model': final_pipeline,
            'score': final_score
        }
        
        print(f'===============  {model_name} test score: {final_score} =============')

    def build_ensemble_cv(self):
        """
        That function builds the ensemble model using models from best optuna tuner cv models and choosen meta model.
        """
        if self.config['dataset']['verbose'] > 0:
            print(f'===============  STARTING BUILD ENSEMBLE FROM CV MODELS  ====================')
        
        if self.config['stacking']['cv_enabled']:

            if self.config['dataset']['verbose'] > 0:
                print(f'Final estimator: {self.config["stacking"]["meta_model"]}')

            for model_name in self.config['models']:
                if self.config['models'][model_name]['enabled']:
                    
                    if self.config['dataset']['verbose'] > 0:
                        print(f'Starting {model_name} cv stacking training ... ', end='')

                    # Disable early stopping for XGBoost models
                    if model_name == 'xgboost':
                        for i in range(len(self.models[model_name]['cv_model'])):
                            # Get the XGBoost model from the pipeline
                            xgb_model = self.models[model_name]['cv_model']['fold' + str(i+1)]['model'].named_steps[model_name]
                            # Disable early stopping
                            xgb_model.set_params(early_stopping_rounds=None)

                    base_estimators = [
                        (model_name + str(i+1), self.models[model_name]['cv_model']['fold' + str(i+1)]['model'])
                        for i in range(len(self.models[model_name]['cv_model']))
                    ]

                    

                    if self.config['dataset']['task_type'] == 'classification':
                        if self.config['stacking']['meta_model'] == 'catboost':
                            final_estimator = CatBoostClassifier(random_state=42, verbose=0, thread_count=self.config['models']['catboost']['params']['thread_count'])
                        elif self.config['stacking']['meta_model'] == 'lightgbm':
                            final_estimator = LGBMClassifier(random_state=42, verbose=0, num_threads=self.config['models']['lightgbm']['params']['num_threads'])
                        elif self.config['stacking']['meta_model'] == 'xgboost':
                            final_estimator = XGBClassifier(random_state=42, verbose=0, nthread=self.config['models']['xgboost']['params']['nthread'])
                        elif self.config['stacking']['meta_model'] == 'linear':
                            final_estimator = RidgeClassifier()
                        
                    elif self.config['dataset']['task_type'] == 'regression':
                        if self.config['stacking']['meta_model'] == 'catboost':
                            final_estimator = CatBoostRegressor(random_state=42, verbose=0, thread_count=self.config['models']['catboost']['params']['thread_count'])
                        elif self.config['stacking']['meta_model'] == 'lightgbm':
                            final_estimator = LGBMRegressor(random_state=42, verbose=0, num_threads=self.config['models']['lightgbm']['params']['num_threads'])
                        elif self.config['stacking']['meta_model'] == 'xgboost':
                            final_estimator = XGBRegressor(random_state=42, verbose=0, nthread=self.config['models']['xgboost']['params']['nthread'])
                        elif self.config['stacking']['meta_model'] == 'linear':
                            final_estimator = Ridge()
                    
                    if self.config['dataset']['task_type'] == 'classification':
                        st = StackingClassifier(
                            estimators=base_estimators,
                            final_estimator=final_estimator,
                            cv=self.config['stacking']['cv_folds'] if self.config['stacking']['prefit'] == False else 'prefit',
                            n_jobs=-1,

                    )
                    elif self.config['dataset']['task_type'] == 'regression':
                        st = StackingRegressor(
                            estimators=base_estimators,
                            final_estimator=final_estimator,
                            cv=self.config['stacking']['cv_folds'] if self.config['stacking']['prefit'] == False else 'prefit',
                            n_jobs=-1,
                        )

                    
                    st.fit(self.X_train, self.y_train)

                    if self.config['dataset']['task_type'] == 'classification':
                        y_pred = st.predict_proba(self.X_test)
                        stacking_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, y_pred[:,1])
                    elif self.config['dataset']['task_type'] == 'regression':
                        y_pred = st.predict(self.X_test)
                        stacking_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, y_pred)
                    
                    self.models[model_name]['ensemble_stacking_cv'] = {
                        'model': st,
                        'score': stacking_score
                    }
                    
                    print(f'score: {self.models[model_name]["ensemble_stacking_cv"]["score"]}')
        
        if self.config['voting']['cv_enabled']:

            for model_name in self.config['models']:
                if self.config['models'][model_name]['enabled']:
                    
                    if self.config['dataset']['verbose'] > 0:
                        print(f'===============  {model_name} voting training ... ', end='')

                    if self.config['voting']['prefit'] == False:
                        base_estimators = [
                            (model_name + str(i+1), self.models[model_name]['cv_model']['fold' + str(i+1)]['model'])
                            for i in range(len(self.models[model_name]['cv_model']))
                        ]
                    else:
                        base_estimators = [
                            self.models[model_name]['cv_model']['fold' + str(i+1)]['model']
                            for i in range(len(self.models[model_name]['cv_model']))
                        ]
                    
                    if self.config['dataset']['task_type'] == 'classification':
                        vt = VotingClassifier(
                            estimators=base_estimators,
                            voting='soft',
                        )
                    elif self.config['dataset']['task_type'] == 'regression':
                        vt = VotingRegressor(
                            estimators=base_estimators,
                        )

                    if self.config['voting']['prefit'] == False:
                        vt.fit(self.X_train, self.y_train)
                    else:
                        vt.estimators_ = base_estimators
                    
                    if self.config['dataset']['task_type'] == 'classification':
                        y_pred = vt.predict_proba(self.X_test)
                        voting_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, y_pred[:,1])
                    elif self.config['dataset']['task_type'] == 'regression':
                        y_pred = vt.predict(self.X_test)
                        voting_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, y_pred)
                    
                    self.models[model_name]['ensemble_voting_cv'] = {
                        'model': vt,
                        'score': voting_score
                    }
                    
                    print(f'score: {self.models[model_name]["ensemble_voting_cv"]["score"]}')

    def build_ensemble_final(self):
        """
        That function builds the final ensemble model using models from best optuna tuner final models and choosen meta model.
        """

        if self.config['dataset']['verbose'] > 0:
            print(f'===============  STARTING BUILD ENSEMBLE FROM FINAL MODELS  ====================')
        
        if self.config['stacking']['final_enabled']:

            base_estimators = []

            for model_name in self.config['models']:
                if self.config['models'][model_name]['enabled']:
                    
                    # Disable early stopping for XGBoost models
                    if model_name == 'xgboost':
                        xgb_model = self.models[model_name]['final_model']['model'].named_steps[model_name]
                        # Disable early stopping
                        xgb_model.set_params(early_stopping_rounds=None)

                    if self.config['dataset']['verbose'] > 0:
                        print(f'Adding {model_name} to base estimators')

                    base_estimators.append(
                        (model_name, self.models[model_name]['final_model']['model'])
                    )


            if self.config['dataset']['task_type'] == 'classification':
                if self.config['stacking']['meta_model'] == 'catboost':
                    final_estimator = CatBoostClassifier(random_state=42, verbose=0, thread_count=self.config['models']['catboost']['params']['thread_count'])
                elif self.config['stacking']['meta_model'] == 'lightgbm':
                    final_estimator = LGBMClassifier(random_state=42, verbose=0, num_threads=self.config['models']['lightgbm']['params']['num_threads'])
                elif self.config['stacking']['meta_model'] == 'xgboost':    
                    final_estimator = XGBClassifier(random_state=42, verbose=0, nthread=self.config['models']['xgboost']['params']['nthread'])
                elif self.config['stacking']['meta_model'] == 'linear':
                    final_estimator = RidgeClassifier()
            elif self.config['dataset']['task_type'] == 'regression':
                if self.config['stacking']['meta_model'] == 'catboost':
                    final_estimator = CatBoostRegressor(random_state=42, verbose=0, thread_count=self.config['models']['catboost']['params']['thread_count'])
                elif self.config['stacking']['meta_model'] == 'lightgbm':
                    final_estimator = LGBMRegressor(random_state=42, verbose=0, num_threads=self.config['models']['lightgbm']['params']['num_threads'])
                elif self.config['stacking']['meta_model'] == 'xgboost':
                    final_estimator = XGBRegressor(random_state=42, verbose=0, nthread=self.config['models']['xgboost']['params']['nthread'])
                elif self.config['stacking']['meta_model'] == 'linear':
                    final_estimator = Ridge()

            if self.config['dataset']['verbose'] > 0:
                print(f'Final estimator: {self.config["stacking"]["meta_model"]}')
                print(f'Starting stacking training ... ', end='')
            
            if self.config['dataset']['task_type'] == 'classification':
                st = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=final_estimator,
                    cv=self.config['stacking']['cv_folds'] if self.config['stacking']['prefit'] == False else 'prefit',
                    n_jobs=-1
                )
            elif self.config['dataset']['task_type'] == 'regression':
                st = StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=final_estimator,
                    cv=self.config['stacking']['cv_folds'] if self.config['stacking']['prefit'] == False else 'prefit',
                    n_jobs=-1
                )
            
            st.fit(self.X_train, self.y_train)

            if self.config['dataset']['task_type'] == 'classification':
                y_pred = st.predict_proba(self.X_test)
                stacking_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, y_pred[:,1])
            elif self.config['dataset']['task_type'] == 'regression':
                y_pred = st.predict(self.X_test)
                stacking_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, y_pred)

            self.models['ensemble_stacking_final'] = {
                    'model': st,
                    'score': stacking_score
                }
            
            if self.config['dataset']['verbose'] > 0:
                print(f'score: {self.models["ensemble_stacking_final"]["score"]}')
        
        if self.config['voting']['final_enabled']:

            base_estimators = []

            for model_name in self.config['models']:
                if self.config['models'][model_name]['enabled']:

                    if self.config['voting']['prefit'] == False:
                        base_estimators.append(
                            (model_name, self.models[model_name]['final_model']['model'])
                        )
                    else:
                        base_estimators.append(
                            self.models[model_name]['final_model']['model']
                        )

            if self.config['dataset']['task_type'] == 'classification':
                vt = VotingClassifier(
                    estimators=base_estimators,
                    voting='soft',
                    n_jobs=-1
                )
            elif self.config['dataset']['task_type'] == 'regression':
                vt = VotingRegressor(
                    estimators=base_estimators,
                    n_jobs=-1
                )
            
            if self.config['dataset']['verbose'] > 0:
                print(f'Starting voting training ... ', end='')

            if self.config['voting']['prefit'] == False:
                vt.fit(self.X_train, self.y_train)
            else:
                vt.estimators_ = base_estimators

            if self.config['dataset']['task_type'] == 'classification':
                y_pred = vt.predict_proba(self.X_test)
                voting_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, y_pred[:,1])
            elif self.config['dataset']['task_type'] == 'regression':
                y_pred = vt.predict(self.X_test)
                voting_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, y_pred)
            
            self.models['ensemble_voting_final'] = {
                'model': vt,
                'score': voting_score
            }
            
            if self.config['dataset']['verbose'] > 0:
                print(f'score: {self.models["ensemble_voting_final"]["score"]}')
