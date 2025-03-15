
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split, GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_log_error, root_mean_squared_error, mean_absolute_percentage_error, accuracy_score, f1_score, roc_auc_score, log_loss, make_scorer
from sklearn.base import clone
import pandas as pd
import numpy as np


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

class psML:
    """
    Machine Learning Pipeline for training, evaluation, and prediction
    """
    def __init__(self, models_config, preprocessor_config, X, y, test_size=0.2, task_type='classification'):
        self.models = {}
        self.models_config = models_config
        self.preprocessor_config = preprocessor_config

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.columns = {}
        self.preprocessor = None

    def optimize(self):
        self.create_preprocessor()
        
        if self.models_config['lightgbm']['enabled']:
            self.optimize_lightgbm_classifier()

        if self.models_config['xgboost']['enabled']:
            self.optimize_xgboost_optimizer()

        if self.models_config['catboost']['enabled']:
            self.optimize_catboost_classifier()
        
        if self.models_config['pytorch']['enabled']:
            self.optimize_pytorch_classifier()
        
    def create_preprocessor(self):
        preprocessor, columns = create_preprocessor(self.preprocessor_config, self.X_train)
        self.preprocessor = preprocessor
        self.columns = columns

    def scores(self):

        if self.models_config['lightgbm']['enabled']:
            print(f'LightGBM CV Score: {self.models["lightgbm"]["cv_score"]}')

        if self.models_config['xgboost']['enabled']:
            print(f'XGBoost CV Score: {self.models["xgboost"]["cv_score"]}')

        if self.models_config['catboost']['enabled']:
            print(f'CatBoost CV Score: {self.models["catboost"]["cv_score"]}')
        
        if self.models_config['pytorch']['enabled']:
            print(f'PyTorch CV Score: {self.models["pytorch"]["cv_score"]}')
        
    
             
    def optimize_lightgbm_classifier(self):
        

        self.models['lightgbm'] = {}
        
        def objective(trial):
            global best_classification_cv
            global best_classification_model

            params = {
                "objective": self.models_config['lightgbm']['params']['objective'],
                "device": self.models_config['lightgbm']['params']['device'],
                "metric": self.models_config['lightgbm']['params']['eval_metric'], 
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
            }

            if params['boosting_type'] != 'goss':
                params["bagging_fraction"] = trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                params["bagging_freq"] =  trial.suggest_int("bagging_freq", 1, 7),
                

            print(f'===============  LightGBM Training - Trail {trial.number}  ==============================')
            print(f'{params}')
            
            split = 5
            scores = 0

            #skf = KFold(n_splits=split, shuffle=True, random_state=42)
            skf = StratifiedKFold(n_splits=split, shuffle=True, random_state=42)

            fitted_trial_models = {}

            for fold, (train_index, val_index) in enumerate(skf.split(self.X_train, self.y_train), 1):
                print(f'Fold {fold} ...', end="")
                
                X_train_fold, X_val_fold = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
                y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]
                
                fold_preprocessor = clone(self.preprocessor)
                fold_preprocessor.fit(X_train_fold, y_train_fold)
                # Transform training and validation data
                X_val_transformed = fold_preprocessor.transform(X_val_fold.reset_index(drop=True))

                pipeline = Pipeline([
                            ('preprocessor', fold_preprocessor),
                            ('lightgbm', LGBMClassifier(**params))  
                        ])
                        
                try:
                    pipeline.fit(
                        X_train_fold.reset_index(drop=True), 
                        y_train_fold.reset_index(drop=True),
                        lightgbm__eval_set=[(X_val_transformed, y_val_fold.reset_index(drop=True))],
                        lightgbm__eval_metric="auc",
                        lightgbm__callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
                        )
                except optuna.exceptions.TrialPruned:
                    print(f"Trial {trial.number} pruned at fold {fold}.")
                    raise  # Re-raise to notify Optuna)
                
                y_pred = pipeline.predict_proba(X_val_fold)
                s = roc_auc_score(y_val_fold.reset_index(drop=True), y_pred[:,1])

                df_y_pred = pd.DataFrame(y_pred[:,1])
                df_y_pred.index = y_val_fold.index
                df_y_pred.columns = [y_val_fold.columns[0]]

                fitted_trial_models[f'fold{fold}'] = {'model': pipeline, 'oof': df_y_pred}
                
                scores += s
                print(f" score: {s}")

                # Report intermediate score
                intermediate_score = np.mean(scores)
                trial.report(intermediate_score, step=fold)

                # Prune trial if necessary
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned after fold {fold}.")
                    raise optuna.exceptions.TrialPruned()
                
            score = scores / split

            if score > best_classification_cv['lightgbm']:
                best_classification_cv['lightgbm'] = score
                best_classification_model['lightgbm'] = fitted_trial_models
            
            print("CV ROC-AUC Score:", score)

            return score

        # Create an Optuna study
        study = optuna.create_study(direction='maximize', study_name='LightGBM Optimization', pruner=pruner, sampler=sampler)

        # Optimize the objective function
        study.optimize(objective, n_trials=self.models_config['lightgbm']['optuna_trials'])

        # Print study statistics
        print("Study statistics:")
        print("  Number of finished trials: ", len(study.trials))
        print("  Best trial:")
        trial = study.best_trial

        print("    Value: {:.4f}".format(trial.value))
        print("    Params: ")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")


        self.models['lightgbm']['cv_score'] = best_classification_cv['lightgbm']
        self.models['lightgbm']['cv_model'] = best_classification_model['lightgbm']
        self.models['lightgbm']['best_params'] = trial.params
        self.models['lightgbm']['study'] = study

        p = trial.params

        params = {
            "objective": "binary",
            "device": self.models_config['lightgbm']['params']['device'],
            "metric": "auc",
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
            }

        if p['boosting_type'] != 'goss':
            params["bagging_fraction"] = p["bagging_fraction"]
            params["bagging_freq"] =  p["bagging_freq"]

        print(params)

        final_preprocessor = clone(self.preprocessor)
        final_preprocessor.fit(self.X_train, self.y_train)
        # Transform training and validation data
        X_test_transformed = final_preprocessor.transform(self.X_test.reset_index(drop=True))

        final_pipeline = Pipeline([
                    ('preprocessor', final_preprocessor),
                    ('lightgbm', LGBMClassifier(**params))  
                ])

        final_pipeline.fit(
            self.X_train.reset_index(drop=True), 
            self.y_train.reset_index(drop=True),
            lightgbm__eval_set=[(X_test_transformed, self.y_test.reset_index(drop=True))],
            lightgbm__eval_metric="auc",
            lightgbm__callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        self.models['lightgbm']['model'] = final_pipeline   
        
    def optimize_xgboost_optimizer(self):

        self.models['xgboost'] = {}
        verbose = 2
        
        def objective(trial):
            global best_classification_cv
            global best_classification_model
            # Suggest hyperparameters
            params = {
                "objective": trial.suggest_categorical('objective', self.models_config['xgboost']['params']['objective']),
                "eval_metric": trial.suggest_categorical('eval_metric', self.models_config['xgboost']['params']['eval_metric']),
                "device": self.models_config['xgboost']['params']['device'],
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
                
            }

            print(f'===============  XGBoost Training - Trail {trial.number}  ==============================')
            print(f'{params}')
            
            split = 5
            scores = 0

            #skf = KFold(n_splits=split, shuffle=True, random_state=42)
            skf = StratifiedKFold(n_splits=split, shuffle=True, random_state=42)
            
            fitted_trial_models = {}

            for fold, (train_index, val_index) in enumerate(skf.split(self.X_train, self.y_train), 1):
                print(f'Fold {fold} ...', end="")
                
                X_train_fold, X_val_fold = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
                y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]
                
                X_train_fold = X_train_fold.reset_index(drop=True)
                y_train_fold = y_train_fold.reset_index(drop=True)

                fold_preprocessor = clone(self.preprocessor)
                fold_preprocessor.fit(X_train_fold, y_train_fold)
                # Transform training and validation data
                X_val_transformed = fold_preprocessor.transform(X_val_fold.reset_index(drop=True))

                pipeline = Pipeline([
                            ('preprocessor', fold_preprocessor),
                            ('xgboost', XGBClassifier(**params))  
                        ])
                        
                try:
                    pipeline.fit(
                        X_train_fold, 
                        y_train_fold,
                        xgboost__eval_set=[(X_val_transformed, y_val_fold.reset_index(drop=True))],
                        xgboost__verbose=False
                        )
                except optuna.exceptions.TrialPruned:
                    print(f"Trial {trial.number} pruned at fold {fold}.")
                    raise  # Re-raise to notify Optuna)
                
                y_pred = pipeline.predict_proba(X_val_fold.reset_index(drop=True))
                s = roc_auc_score(y_val_fold.reset_index(drop=True), y_pred[:,1])

                df_y_pred = pd.DataFrame(y_pred[:,1])
                df_y_pred.index = y_val_fold.index
                df_y_pred.columns = [y_val_fold.columns[0]]

                fitted_trial_models[f'fold{fold}'] = {'model': pipeline, 'oof': df_y_pred}

                scores += s
                print(f" score: {s}")

                # Report intermediate score
                intermediate_score = np.mean(scores)
                trial.report(intermediate_score, step=fold)

                # Prune trial if necessary
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned after fold {fold}.")
                    raise optuna.exceptions.TrialPruned()
            
            score = scores / split

            if score > best_classification_cv['xgboost']:
                best_classification_cv['xgboost'] = score
                best_classification_model['xgboost'] = fitted_trial_models

            print(f"CV ROC-AUC Score: {score:.5f}")

            return score


        # Create an Optuna study
        study = optuna.create_study(direction='maximize', study_name='XGBoost Optimization', pruner=pruner, sampler=sampler)

        # Optimize the objective function
        study.optimize(objective, n_trials=self.models_config['xgboost']['optuna_trials'])

        # Print study statistics
        print("Study statistics:")
        print("  Number of finished trials: ", len(study.trials))
        print("  Best trial:")
        trial = study.best_trial

        print("    Value: {:.4f}".format(trial.value))
        print("    Params: ")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")


        self.models['xgboost']['cv_score'] = best_classification_cv['xgboost']
        self.models['xgboost']['cv_model'] = best_classification_model['xgboost']
        self.models['xgboost']['best_params'] = trial.params
        self.models['xgboost']['study'] = study

        p = trial.params

        params = {
            "objective": p['objective'],
            "eval_metric": p['eval_metric'],
            "device": self.models_config['xgboost']['params']['device'],
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
            "colsample_bylevel": p['colsample_bylevel']
        }

        print(params)

        final_preprocessor = clone(self.preprocessor)
        final_preprocessor.fit(self.X_train, self.y_train)
        # Transform training and validation data
        X_test_transformed = final_preprocessor.transform(self.X_test.reset_index(drop=True))

        final_pipeline = Pipeline([
                    ('preprocessor', final_preprocessor),
                    ('xgboost', XGBClassifier(**params))  
                ])

        final_pipeline.fit(
            self.X_train.reset_index(drop=True), 
            self.y_train.reset_index(drop=True), 
            xgboost__eval_set=[(X_test_transformed, self.y_test.reset_index(drop=True))],
            xgboost__verbose=False
        )
        
        self.models['xgboost']['model'] = final_pipeline   

    def optimize_catboost_classifier(self):
        self.models['catboost'] = {}
        verbose = 2
        
        def objective(trial):
            global best_classification_cv
            global best_classification_model
            # Suggest hyperparameters
            params = {
                'loss_function': trial.suggest_categorical('loss_function', self.models_config['catboost']['params']['objective']),
                'eval_metric': trial.suggest_categorical('eval_metric', self.models_config['catboost']['params']['eval_metric']),
                'task_type': 'CPU' if self.models_config['catboost']['params']['device'] == 'cpu' else 'GPU',
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
            }
            if params['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
            if params['grow_policy'] == 'Lossguide':
                params['max_leaves'] = trial.suggest_int('max_leaves', 2, 32)

            print(f'===============  CatBoost Training - Trail {trial.number}  ==============================')
            print(f'{params}')
            
            split = 5
            scores = 0

            #skf = KFold(n_splits=split, shuffle=True, random_state=42)
            skf = StratifiedKFold(n_splits=split, shuffle=True, random_state=42)
            
            fitted_trial_models = {}

            for fold, (train_index, val_index) in enumerate(skf.split(self.X_train, self.y_train), 1):
                print(f'Fold {fold} ...', end="")
                
                X_train_fold, X_val_fold = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
                y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]
                
                X_train_fold = X_train_fold.reset_index(drop=True)
                y_train_fold = y_train_fold.reset_index(drop=True)

                fold_preprocessor = clone(self.preprocessor)
                fold_preprocessor.fit(X_train_fold, y_train_fold)
                # Transform training and validation data
                X_val_transformed = fold_preprocessor.transform(X_val_fold.reset_index(drop=True))

                pipeline = Pipeline([
                            ('preprocessor', fold_preprocessor),
                            ('catboost', CatBoostClassifier(**params))  
                        ])
                        
                try:
                    pipeline.fit(
                        X_train_fold, 
                        y_train_fold,
                        catboost__eval_set=[(X_val_transformed, y_val_fold.reset_index(drop=True))],
                        catboost__early_stopping_rounds=100, 
                        )
                except optuna.exceptions.TrialPruned:
                    print(f"Trial {trial.number} pruned at fold {fold}.")
                    raise  # Re-raise to notify Optuna)
                
                y_pred = pipeline.predict_proba(X_val_fold.reset_index(drop=True))
                s = roc_auc_score(y_val_fold.reset_index(drop=True), y_pred[:,1])

                df_y_pred = pd.DataFrame(y_pred[:,1])
                df_y_pred.index = y_val_fold.index
                df_y_pred.columns = [y_val_fold.columns[0]]

                fitted_trial_models[f'fold{fold}'] = {'model': pipeline, 'oof': df_y_pred}
                
                scores += s
                print(f" score: {s}")

                # Report intermediate score
                intermediate_score = np.mean(scores)
                trial.report(intermediate_score, step=fold)

                # Prune trial if necessary
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned after fold {fold}.")
                    raise optuna.exceptions.TrialPruned()
            
            score = scores / split

            if score > best_classification_cv['catboost']:
                best_classification_cv['catboost'] = score
                best_classification_model['catboost'] = fitted_trial_models

            print("CV ROC-AUC Score:", score)

            return score

        # Create an Optuna study
        study = optuna.create_study(direction='maximize', study_name='Catboost Optimization', pruner=pruner, sampler=sampler)

        # Optimize the objective function
        study.optimize(objective, n_trials=self.models_config['catboost']['optuna_trials'])

        # Print study statistics
        print("Study statistics:")
        print("  Number of finished trials: ", len(study.trials))
        print("  Best trial:")
        trial = study.best_trial

        print("    Value: {:.4f}".format(trial.value))
        print("    Params: ")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")

        self.models['catboost']['cv_score'] = best_classification_cv['catboost']
        self.models['catboost']['cv_model'] = best_classification_model['catboost']
        self.models['catboost']['best_params'] = trial.params
        self.models['catboost']['study'] = study

        p = dict(trial.params.items())
        params = {
            'loss_function': p['loss_function'],
            'eval_metric': p['eval_metric'],
            'task_type': 'CPU' if self.models_config['catboost']['params']['device'] == 'cpu' else 'GPU',
            'n_estimators': p['n_estimators'],
            'learning_rate': p['learning_rate'],
            'depth': p['depth'],
            'l2_leaf_reg': p['l2_leaf_reg'],
            'border_count': p['border_count'],
            'random_seed': 42,
            'verbose': False,
            'bootstrap_type': p['bootstrap_type'],
            #'boosting_type': p['boosting_type'],
            'feature_border_type': p['feature_border_type'],
            'leaf_estimation_iterations': p['leaf_estimation_iterations'],
            'min_data_in_leaf': p['min_data_in_leaf'],
            'random_strength': p['random_strength'],
            'grow_policy': p['grow_policy'],
            }

        if p['bootstrap_type'] == 'Bernoulli':
            params['subsample'] = p['subsample']
        if p['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = p['bagging_temperature']
        if p['grow_policy'] == 'Lossguide':
                params['max_leaves'] = p['max_leaves']

        print(params)  

        final_preprocessor = clone(self.preprocessor)
        final_preprocessor.fit(self.X_train, self.y_train)
        # Transform training and validation data
        X_test_transformed = final_preprocessor.transform(self.X_test.reset_index(drop=True))

        final_pipeline = Pipeline([
                    ('preprocessor', final_preprocessor),
                    ('catboost', CatBoostClassifier(**params))  
                ])

        final_pipeline.fit(
            self.X_train, 
            self.y_train,
            catboost__eval_set=[(X_test_transformed, self.y_test.reset_index(drop=True))],
            catboost__early_stopping_rounds=100, 
        )
        
        self.models['catboost']['model'] = final_pipeline

    def optimize_pytorch_classifier(self):
        self.models['pytorch'] = {}
        verbose = 2
        
        def objective(trial):
            global best_classification_cv
            global best_classification_model

            # Suggest hyperparameters
            params = {
                "learning_rate": trial.suggest_categorical('learning_rate', self.models_config['pytorch']['params']['learning_rate']),
                "optimizer_name": trial.suggest_categorical('optimizer_name', self.models_config['pytorch']['params']['optimizer_name']),
                "batch_size": trial.suggest_categorical('batch_size', self.models_config['pytorch']['params']['batch_size']),
                "net":  trial.suggest_categorical('net', self.models_config['pytorch']['params']['net']),
                "weight_init": trial.suggest_categorical('weight_init', [
                    'default',
                    #'xavier_normal',
                    'kaiming_normal', #Relu
                    #'xavier_uniform',
                    'kaiming_uniform', #Relu
                ]),

                "verbose": 1,
                "device": 'cuda' if self.models_config['pytorch']['params']['device'] == 'gpu' else 'cpu',
                "max_epochs": self.models_config['pytorch']['params']['train_max_epochs'],
                "patience": self.models_config['pytorch']['params']['train_patience'],
                "loss": self.models_config['pytorch']['params']['objective'],
                "embedding_info": self.columns['embedding_info'],
            }

            print(f'===============  PyTorch Training - Trail {trial.number}  =============================')
            print(f'{params}')
            
            split = 5
            scores = 0

            #skf = KFold(n_splits=split, shuffle=True, random_state=42)
            skf = StratifiedKFold(n_splits=split, shuffle=True, random_state=42)
            

            fitted_trial_models = {}

            for fold, (train_index, val_index) in enumerate(skf.split(self.X_train, self.y_train), 1):
                print(f'Fold {fold} ...')
                
                X_train_fold, X_val_fold = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
                y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]
                
                fold_preprocessor = clone(self.preprocessor)
                fold_preprocessor.fit(X_train_fold, y_train_fold)
                # Transform training and validation data
                X_val_transformed = fold_preprocessor.transform(X_val_fold.reset_index(drop=True))

                pipeline = Pipeline([
                            ('preprocessor', fold_preprocessor),
                            ('torch', PyTorchClassifier(**params))  
                        ])
                        
                try:
                    pipeline.fit(
                        X_train_fold, 
                        y_train_fold,
                        torch__eval_set = [X_val_transformed, y_val_fold]
                        )
                except optuna.exceptions.TrialPruned:
                    print(f"Trial {trial.number} pruned at fold {fold}.")
                    raise  # Re-raise to notify Optuna)

                
                #y_pred = pipeline.predict(X_test_fold.reset_index(drop=True))
                y_pred = pipeline.predict_proba(X_val_fold)
                s = roc_auc_score(y_val_fold, y_pred[:,1])

                df_y_pred = pd.DataFrame(y_pred[:,1])
                df_y_pred.index = y_val_fold.index
                df_y_pred.columns = [y_val_fold.columns[0]]

                fitted_trial_models[f'fold{fold}'] = {'model': pipeline, 'oof': df_y_pred}
                
                scores += s
                print(f"Fold {fold} score: {s}")

                # Report intermediate score
                intermediate_score = np.mean(scores)
                trial.report(intermediate_score, step=fold)

                # Prune trial if necessary
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned after fold {fold}.")
                    raise optuna.exceptions.TrialPruned()
            
            score = scores / split

            if score > best_classification_cv['pytorch']:
                best_classification_cv['pytorch'] = score
                best_classification_model['pytorch'] = fitted_trial_models
                

            print("CV ROC-AUC Score:", score)

            return score

        
        # Create an Optuna study
        study = optuna.create_study(direction='maximize', study_name='PyTorch Optimization', pruner=pruner, sampler=sampler)

        # Optimize the objective function
        study.optimize(objective, n_trials=self.models_config['pytorch']['optuna_trials'])

        # Print study statistics
        print("Study statistics:")
        print("  Number of finished trials: ", len(study.trials))
        print("  Best trial:")
        trial = study.best_trial

        print("    Value: {:.4f}".format(trial.value))
        print("    Params: ")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")

        

        self.models['pytorch']['cv_score'] = best_classification_cv['pytorch']
        self.models['pytorch']['cv_model'] = best_classification_model['pytorch']
        self.models['pytorch']['best_params'] = trial.params
        self.models['pytorch']['study'] = study

        p = trial.params

        params = {
            "learning_rate": p["learning_rate"],
            "optimizer_name": p["optimizer_name"],
            "batch_size": p["batch_size"],
            "weight_init": p["weight_init"],
            "max_epochs": self.models_config['pytorch']['params']['final_max_epochs'],
            "patience":  self.models_config['pytorch']['params']['final_patience'],
            "net": p['net'],
            "verbose": 1,
            "device": 'cuda' if self.models_config['pytorch']['params']['device'] == 'gpu' else 'cpu',
            "loss":  self.models_config['pytorch']['params']['objective'],
            "embedding_info": self.columns['embedding_info'],
            }

        final_preprocessor = clone(self.preprocessor)
        final_preprocessor.fit(self.X_train, self.y_train)
        # Transform training and validation data
        X_test_transformed = final_preprocessor.transform(self.X_test)

        final_pipeline = Pipeline([
                    ('preprocessor', final_preprocessor),
                    ('torch', PyTorchClassifier(**params))  
                ])
                        
        final_pipeline.fit(
            self.X_train, 
            self.y_train, 
            torch__eval_set=[X_test_transformed, self.y_test])
        
        self.models['pytorch']['model'] = final_pipeline   