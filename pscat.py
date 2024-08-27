import time, random
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold
import catboost as cat

import matplotlib.pyplot as plt
import seaborn as sns
palette = ["#365486", "#7FC7D9", "#0F1035", "#7A316F", "#7A316F"]
sns.set_palette(palette=palette)

warnings.filterwarnings("ignore")

class psCAT:
    model = None
    params = None
    randSeed = 123,
    task_type = 'GPU'
    num_boost_round_param = [10000]
    eta_param = [0.01, 0.05, 0.1, 0.2, 0.3]
    max_depth_param = [3,5,7,10]
    lambda_param = [3]
    objective_param = ['RMSE', 'MAE', 'MAPE']
    subsample_param = [0.8]
    early_stopping_rounds_param = [10]
    grow_policy_param = ['Depthwise']      # ['SymmetricTree', 'Depthwise', 'Lossguide']       
    eval_metric_param = ['RMSE', 'MAE', 'MAPE'] # ['RMSE', 'MAE', 'MAPE', 'R2']
    bootstrap_type = 'Poisson'
    cat_features = []
    metric_out = "rmse" # ['rmse', 'mae', 'mape', 'r2', 'accuracy', 'precision', 'recall', 'f1']
    cv = 5
    build_test_size = 0.2
    
    def __init__(self, all_params=None):
        if all_params is not None:
            for key, value in all_params.items():
                setattr(self, key, value)
    
    
    def buildRegressor(self, X , y):
                
        fitted_models = {}
        
        modelCount = 1
        
        # Calculate all possible combinations and time remaining
        allVarsCount = len(self.num_boost_round_param) * len(self.eta_param) * len(self.max_depth_param) * len(self.lambda_param) * len(self.objective_param) * len(self.subsample_param) * len(self.early_stopping_rounds_param) * len(self.grow_policy_param)
        buildTime = time.time()
        
        print("============= Training started Catboost =============")
        print("Try to build ", allVarsCount, " models in ", self.cv, " fold(s)")
        print("===========================================================")

        for num_boost_round in self.num_boost_round_param:
            for eta in self.eta_param:
                for max_depth in self.max_depth_param:
                    for lambdap in self.lambda_param:
                        for objective in self.objective_param:
                            for subsample in self.subsample_param:
                                for early_stopping_rounds in self.early_stopping_rounds_param:
                                    for grow_policy in self.grow_policy_param:
                                        for eval_metric in self.eval_metric_param:
                                        
                                            # Train model
                                            ts = time.time()
                                            
                                            cv_models = {}
                                                              
                                            if(self.cv > 1):
                                                  
                                                kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.randSeed)
                                                                    
                                                i = 1
                                                for train_index, test_index in kf.split(X):
                                                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                                                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                                                                        
                                                    model1 = cat.CatBoostRegressor(random_state = self.randSeed,
                                                                                task_type = self.task_type,
                                                                                num_boost_round=num_boost_round,
                                                                                eta=eta,
                                                                                max_depth=max_depth,
                                                                                l2_leaf_reg=lambdap,
                                                                                objective=objective,
                                                                                subsample=subsample,
                                                                                early_stopping_rounds=early_stopping_rounds,         
                                                                                grow_policy=grow_policy,
                                                                                eval_metric=eval_metric,
                                                                                bootstrap_type=self.bootstrap_type
                                                                                )
                                                    
                                                    model1.fit(X_train, y_train, verbose=1000,cat_features=self.cat_features, plot=False, eval_set=(X_test, y_test))
                                                
                                                

                                                preds = model1.predict(X_test)
                                                rmse = mean_squared_error(y_test, preds, squared=False)
                                                mae = mean_absolute_error(y_test, preds)
                                                r2 = r2_score(y_test, preds)
                                                mape = mean_absolute_percentage_error(y_test, preds)
                                                medae = median_absolute_error(y_test, preds)
                                                
                                                cv_models[i] = {"rmse":rmse, "mae":mae, "mape": mape, "medae": medae,"r2":r2,  "model": model1}
                                                i += 1
                                                
                                            elif(self.cv == 1):
                                                                
                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.build_test_size, random_state=self.randSeed)
                                                
                                                model1 = cat.CatBoostRegressor(random_state = self.randSeed,
                                                                                task_type = self.task_type,
                                                                                num_boost_round=num_boost_round,
                                                                                eta=eta,
                                                                                max_depth=max_depth,
                                                                                l2_leaf_reg=lambdap,
                                                                                objective=objective,
                                                                                subsample=subsample,
                                                                                early_stopping_rounds=early_stopping_rounds,         
                                                                                grow_policy=grow_policy,
                                                                                eval_metric=eval_metric,
                                                                                bootstrap_type=self.bootstrap_type
                                                                                )
                                                    
                                                model1.fit(X_train, y_train, verbose=1000,cat_features=self.cat_features, plot=False, eval_set=(X_test, y_test))
                                                
                                                

                                                preds = model1.predict(X)
                                                rmse = mean_squared_error(y, preds, squared=False)
                                                mae = mean_absolute_error(y, preds)
                                                r2 = r2_score(y, preds)
                                                mape = mean_absolute_percentage_error(y, preds)
                                                medae = median_absolute_error(y, preds)
                                                
                                                cv_models[1] = {"rmse":rmse, "mae":mae, "mape": mape, "medae": medae,"r2":r2,  "model": model1}
                                                                 
                                            
                                            dfcv = pd.DataFrame(cv_models).T
                                                                
                                            rmse = dfcv['rmse'].mean()
                                            mae = dfcv['mae'].mean()
                                            mape = dfcv['mape'].mean()
                                            medae = dfcv['medae'].mean()
                                            r2 = dfcv['r2'].mean()
                                            
                                            cv_best_model = dfcv[dfcv[self.metric_out] == dfcv[self.metric_out].min()]
                                            
                                            tt = time.time() - ts
                                            
                                            avgTime = (time.time() - buildTime) / modelCount
                                            timeRemaining = (allVarsCount - modelCount) * avgTime
                                            tir = "s"
                                            
                                            if(timeRemaining > 3600):
                                                timeRemaining = timeRemaining / 3600
                                                tir = "h"
                                            elif(timeRemaining > 60):
                                                timeRemaining = timeRemaining / 60
                                                tir = "m"
                                            
                                            fitted_models[modelCount] = {"rmse":rmse, "mae":mae, "mape": mape, "medae":medae, "r2":r2, 
                                                                            "objective":objective,
                                                                            "early_stopping_rounds":early_stopping_rounds,
                                                                            "eta":eta,
                                                                            "max_depth":max_depth,
                                                                            "lambda":lambdap,
                                                                            "eval_metric":eval_metric,
                                                                            "time":tt,
                                                                            "model": cv_best_model['model'].values[0]
                                                                            }
                                            print(modelCount,"/",allVarsCount, " ({tt:.2f}s ~ {tr:.2f}{tir}) ===> ".format(tt=tt,tr=timeRemaining, tir=tir),fitted_models[modelCount], "\n")
                                            
                                            modelCount += 1

        df = pd.DataFrame(fitted_models).T
        best_model = df[df[self.metric_out] == df[self.metric_out].min()]
        
        atir = "s"
        atime = time.time() - buildTime
        if(atime > 3600):
            atime = atime / 3600
            atir = "h"
        elif(atime > 60):
            atime = atime / 60
            atir = "m"
        print("============= Training completed =============")
        print("Trained ", modelCount-1, " models in ", atime, atir) 
        print("============= Best avg metrics ================")
        print("RMSE: ", best_model['rmse'].values[0], "MAE: ", best_model['mae'].values[0], "MAPE: ", best_model['mape'].values[0], "MedAE: ", best_model['medae'].values[0], "R2: ", best_model['r2'].values[0])
        
        preds = best_model['model'].values[0].predict(X)
        rmse = mean_squared_error(y, preds, squared=False)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        mape = mean_absolute_percentage_error(y, preds)
        medae = median_absolute_error(y, preds)
        
        print("============= Best model ================")
        print("RMSE: ", rmse, "MAE: ", mae, "MAPE: ", mape, "MedAE: ", medae, "R2: ", r2)
        
        dfA = y.copy()
        varName = dfA.columns[0]
        dfA.rename(columns = {varName:'Actual'}, inplace = True) 
        dfA['Predicted'] = preds
        
        plt.figure(figsize=(8, 5))
        sns.regplot(data=dfA, x="Actual", y="Predicted")
        plt.show()
        plt.clf()
    
        self.model = best_model['model'].values[0]
        self.params = best_model

    def buildClassifier(self, X , y):
        
        fitted_models = {}
        
        modelCount = 1
        
        # Calculate all possible combinations and time remaining
        allVarsCount = len(self.num_boost_round_param) * len(self.eta_param) * len(self.max_depth_param) * len(self.lambda_param) * len(self.objective_param) * len( self.subsample_param) * len(self.early_stopping_rounds_param) * len(self.grow_policy_param)
        buildTime = time.time()
        
        print("============= Training started Catboost =============")
        print("Try to build ", allVarsCount, " models") 
        print("=====================================================")

        for num_boost_round in self.num_boost_round_param:
            for eta in self.eta_param:
                for max_depth in self.max_depth_param:
                    for lambdap in self.lambda_param:
                        for objective in self.objective_param:
                            for subsample in self.subsample_param:
                                for early_stopping_rounds in self.early_stopping_rounds_param:
                                    for grow_policy in self.grow_policy_param:
                                        for eval_metric in self.eval_metric_param:
                                        
                                            # Train model
                                            ts = time.time()
                                            
                                            cv_models = {}
                                            
                                            if(self.cv > 1):
                                                                
                                                kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.randSeed)
                                                                    
                                                i = 1
                                                for train_index, test_index in kf.split(X):
                                                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                                                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                                                
                                                    model1 = cat.CatBoostClassifier(random_state = self.randSeed,task_type = self.task_type,num_boost_round=num_boost_round,eta=eta,max_depth=max_depth,l2_leaf_reg=lambdap,objective=objective,subsample=subsample,early_stopping_rounds=early_stopping_rounds,grow_policy=grow_policy,eval_metric=eval_metric,bootstrap_type=self.bootstrap_type)
                                                    model1.fit(X_train, y_train, verbose=1000,cat_features=self.cat_features, plot=False, eval_set=(X_test, y_test))
                                                    
                                                    preds = model1.predict(X_test)
                                                    accuracy = accuracy_score(y_test, preds)
                                                    precision = precision_score(preds,y_test, average='weighted')
                                                    recall = recall_score(preds,y_test, average='weighted')
                                                    f1 = f1_score(preds,y_test, average='weighted')
                                                    
                                                    cv_models[i] = {"accuracy":accuracy, "precision":precision, "recall": recall, "f1":f1, "model": model1}
                                                    i += 1
                                            elif(self.cv == 1):
                                                
                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.build_test_size, random_state=self.randSeed)
                                                
                                                model1 = cat.CatBoostClassifier(random_state = self.randSeed,task_type = self.task_type,num_boost_round=num_boost_round,eta=eta,max_depth=max_depth,l2_leaf_reg=lambdap,objective=objective,subsample=subsample,early_stopping_rounds=early_stopping_rounds,grow_policy=grow_policy,eval_metric=eval_metric,bootstrap_type=self.bootstrap_type)
                                                model1.fit(X, y, verbose=1000,cat_features=self.cat_features, plot=False, eval_set=(X_test, y_test))
                                                
                                                preds = model1.predict(X)
                                                accuracy = accuracy_score(y, preds)
                                                precision = precision_score(preds,y, average='weighted')
                                                recall = recall_score(preds,y, average='weighted')
                                                f1 = f1_score(preds,y, average='weighted')
                                                
                                                cv_models[1] = {"accuracy":accuracy, "precision":precision, "recall": recall, "f1":f1, "model": model1}

                                            
                                            dfcv = pd.DataFrame(cv_models).T
                                                                
                                            accuracy = dfcv['accuracy'].mean()
                                            precision = dfcv['precision'].mean()
                                            recall = dfcv['recall'].mean()
                                            f1 = dfcv['f1'].mean()
                                            
                                            cv_best_model = dfcv[dfcv[self.metric_out] == dfcv[self.metric_out].max()]
                                            
                                            tt = time.time() - ts
                                            
                                            avgTime = (time.time() - buildTime) / modelCount
                                            timeRemaining = (allVarsCount - modelCount) * avgTime
                                            tir = "s"
                                            
                                            if(timeRemaining > 3600):
                                                timeRemaining = timeRemaining / 3600
                                                tir = "h"
                                            elif(timeRemaining > 60):
                                                timeRemaining = timeRemaining / 60
                                                tir = "m"
                                            
                                            fitted_models[modelCount] = {"accuracy":accuracy, "precision":precision, "recall": recall, "f1":f1, 
                                                                            "objective":objective,
                                                                            "early_stopping_rounds":early_stopping_rounds,
                                                                            "eta":eta,
                                                                            "max_depth":max_depth,
                                                                            "lambda":lambdap,
                                                                            "eval_metric":eval_metric,
                                                                            "grow_policy":grow_policy,
                                                                            "subsample":subsample,
                                                                            "task_type":self.task_type,
                                                                            "time":tt,
                                                                            "model": cv_best_model['model'].values[0]
                                                                            }
                                            print(modelCount,"/",allVarsCount, " ({tt:.2f}s ~ {tr:.2f}{tir}) ===> ".format(tt=tt,tr=timeRemaining, tir=tir),fitted_models[modelCount], "\n")
                                            
                                            modelCount += 1

        df = pd.DataFrame(fitted_models).T
        best_model = df[df[self.metric_out] == df[self.metric_out].max()]
        
        atir = "s"
        atime = time.time() - buildTime
        if(atime > 3600):
            atime = atime / 3600
            atir = "h"
        elif(atime > 60):
            atime = atime / 60
            atir = "m"
        print("============= Training completed =============")
        print("Trained ", modelCount-1, " models in ", atime, atir) 
        print("============= Best avg metrics ================")
        print("Accuracy: ", best_model['accuracy'].values[0], "Precision: ", best_model['precision'].values[0], "Recall: ", best_model['recall'].values[0], "F1: ", best_model['f1'].values[0])
        print("============= Best model report ===============")
        preds = best_model['model'].values[0].predict(X)
        print(classification_report(y, preds))
        
        cm = confusion_matrix(y, preds, normalize='true')
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        disp.ax_.set_title("Confusion Matrix for CatBoost")
    
        self.model = best_model['model'].values[0]
        self.params = best_model
    
    def predict(self,X):
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)    