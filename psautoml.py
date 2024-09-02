import seaborn as sns
palette = ["#7FC7D9", "#7A316F", "#365486" , "#0F1035", "#7A316F"]
sns.set_palette(palette=palette)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import psai.psviz as psviz
import psai.psxgb as psxgb
import psai.pscat as pscat
import psai.pstf as pstf
import psai.psout as psout
import math as math
import time
from datetime import datetime
#from numba import cuda 

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import tensorflow as tf
from tensorflow import keras


class psAUTOML:
    #zmienne dla treningu z cv
    xgbcv = None
    catcv = None
    tfcv = None
    
    #zmienne dla modelu koncowego
    xgb = None
    cat = None
    tf= None
    
    estimators = None
    task = None
    
    xgbparamscv = None
    catparamscv = None
    tfparamscv = None
    
    
    def __init__(self, task = None, estimators = ['xgb','cat','tf'], xgbparams = None, catparams = None, tfparams = None):
        
        self.task = task
        self.estimators = estimators
        self.xgbparamscv = xgbparams
        self.catparamscv = catparams
        self.tfparamscv = tfparams
        
        
    def fit(self, X, y):
        
        print("============== Training started for PSAUTOML ===============")
        print("Estimators: ", self.estimators, " Task: ", self.task)
        
        ns = datetime.now()
        ts = time.time()
        
        print("============================================================")
        print("Time: ", ns.strftime("%d/%m/%Y %H:%M:%S"), "")
        
        if self.task == 'classification':

            for e in self.estimators:
                if e == 'xgb':
                    self.xgbcv = psxgb.psXGB(all_params=self.xgbparamscv)
                    self.xgbcv.buildClassifier(X,y)
                elif e == 'cat':
                    self.catcv = pscat.psCAT(all_params=self.catparamscv)
                    self.catcv.buildClassifier(X,y)
                elif e == 'tf':
                    self.tfcv = pstf.psTF(all_params=self.tfparamscv)
                    self.tfcv.buildClassifier(X,y)
                else:
                    print('Estimator not recognized')
                
                

        
        if self.task == 'regression':

            for e in self.estimators:
                if e == 'xgb':
                    self.xgbcv = psxgb.psXGB(all_params=self.xgbparamscv)
                    self.xgbcv.buildRegressor(X,y)
                elif e == 'cat':
                    self.catcv = pscat.psCAT(all_params=self.catparamscv)
                    self.catcv.buildRegressor(X,y)
                elif e == 'tf':
                    self.tfcv = pstf.psTF(all_params=self.tfparamscv)
                    self.tfcv.buildRegressor(X,y)
                else:
                    print('Estimator not recognized')
                

        
        nt = datetime.now()
        tt = time.time()
        
        atir = "s"
        atime = tt - ts
        if(atime > 3600):
            atime = atime / 3600
            atir = "h"
        elif(atime > 60):
            atime = atime / 60
            atir = "m"
        
        print("============== Training complete============================")
        print("Estimators: ", self.estimators, " Task: ", self.task)
        print("============================================================")
        print("Time: ", nt.strftime("%d/%m/%Y %H:%M:%S"), ", it takes: ", atime, atir )
        
        self.copyBestParams()
        
    
    def copyBestParams(self):
        if self.xgbcv != None:
            ind = self.xgbcv.params.index[0]
            self.xgbparams = {
                'cv' : 1,
                'build_test_size': self.xgbparamscv['build_test_size'], 
                'randSeed': self.xgbparamscv['randSeed'],
                'tree_method_param': self.xgbparamscv['tree_method_param'],
                'objective_param': [self.xgbcv.params['objective'][ind]],
                'max_depth': [self.xgbcv.params['max_depth'][ind]],
                'eval_metric_param': self.xgbparamscv['eval_metric_param'],  
                'early_stopping_rounds_param': [self.xgbcv.params['early_stopping_rounds'][ind]],
                'num_boost_round': 10000,
                'lambda_param': [self.xgbcv.params['lambda'][ind]],
                'alpha_param': [self.xgbcv.params['alpha'][ind]],
                'gamma_param': [self.xgbcv.params['gamma'][ind]],
                'min_child_weight_param': [self.xgbcv.params['min_child_weight'][ind]],
                'subsample_param': [self.xgbcv.params['subsample'][ind]],
                'eta_param': [self.xgbcv.params['eta'][ind]],
                'colsample_bytree_param': [self.xgbcv.params['colsample_bytree'][ind]],
                'num_class':  self.xgbparamscv['num_class'] if self.task == 'classification' else 0,
                'metric_out':  self.xgbparamscv['metric_out']
                }
        if self.catcv != None:
            ind = self.catcv.params.index[0]
            self.catparams = {
                'cv' : 1,
                'randSeed' : self.catparamscv['randSeed'],
                'build_test_size': self.catparamscv['build_test_size'],
                'task_type' : self.catparamscv['task_type'],
                'max_depth_param' : [self.catcv.params['max_depth'][ind]],
                'eval_metric_param' : [self.catcv.params['eval_metric'][ind]],  
                'early_stopping_rounds_param' : [self.catcv.params['early_stopping_rounds'][ind]],
                'objective_param' : [self.catcv.params['objective'][ind]],  
                'lambda_param' : [self.catcv.params['lambda'][ind]], 
                'subsample_param' : [self.catcv.params['subsample'][ind]], 
                'eta_param' : [self.catcv.params['eta'][ind]], 
                'grow_policy_param' : [self.catcv.params['grow_policy'][ind]],
                'bootstrap_type' : self.catcv.params['bootstrap_type'][ind],
                'metric_out' :  self.catparamscv['metric_out']
                }
        if self.tfcv != None:
            ind = self.tfcv.params.index[0]
            
            layers = {
                "nn1": self.tfcv.params['layers'][ind]
            }

            self.tfparams = {
                'cv' : 1,
                'randSeed': self.tfparamscv['randSeed'],
                'build_test_size': self.tfparamscv['build_test_size'],
                'epochs_param': [1000],
                'optimizer_param': [self.tfcv.params['optimizer'][ind]],
                'loss_param': [self.tfcv.params['loss'][ind]],
                'batch_size_param': [self.tfcv.params['batch_size'][ind]],
                'early_stopping_rounds_param': [self.tfcv.params['early_stopping_rounds'][ind]],
                'layers_param': layers,
                'min_lr': self.tfcv.params['min_lr'][ind],
                'metrics': self.tfparamscv['metrics'],
                'metric_out': self.tfparamscv['metric_out']
                }
    
                    
    def build(self, X, y):
        
        print("============== Training started for PSAUTOML =================")
        print("Estimators: ", self.estimators, " Task: ", self.task)
        
        ns = datetime.now()
        ts = time.time()
        
        print("==============================================================")
        print("Time: ", ns.strftime("%d/%m/%Y %H:%M:%S"), "")
        
        
        if self.task == 'classification':

            for e in self.estimators:
                if e == 'xgb':
                    self.xgbparams['cv'] = 1
                    self.xgb = psxgb.psXGB(all_params=self.xgbparams)
                    self.xgb.buildClassifier(X,y)
                elif e == 'cat':
                    self.catparams['cv'] = 1
                    self.cat = pscat.psCAT(all_params=self.catparams)
                    self.cat.buildClassifier(X,y)
                elif e == 'tf':
                    self.tfparams['cv'] = 1
                    self.tf = pstf.psTF(all_params=self.tfparams)
                    self.tf.buildClassifier(X,y)
                else:
                    print('Estimator not recognized')
                

        
        if self.task == 'regression':

            for e in self.estimators:
                if e == 'xgb':
                    self.xgbparams['cv'] = 1
                    self.xgb = psxgb.psXGB(all_params=self.xgbparams)
                    self.xgb.buildRegressor(X,y)
                elif e == 'cat':
                    self.catparams['cv'] = 1
                    self.cat = pscat.psCAT(all_params=self.catparams)
                    self.cat.buildRegressor(X,y)
                elif e == 'tf':
                    self.tfparams['cv'] = 1
                    self.tf = pstf.psTF(all_params=self.tfparams)
                    self.tf.buildRegressor(X,y)
                else:
                    print('Estimator not recognized')
                
                    
        nt = datetime.now()
        tt = time.time()
        
        atir = "s"
        atime = tt - ts
        if(atime > 3600):
            atime = atime / 3600
            atir = "h"
        elif(atime > 60):
            atime = atime / 60
            atir = "m"
        
        print("============== Training complete=============================")
        print("Estimators: ", self.estimators, " Task: ", self.task)
        print("=============================================================")
        print("Time: ", nt.strftime("%d/%m/%Y %H:%M:%S"), ", it takes: ", atime, atir )
                    
    def evaluate(self, X, y):
        
        if self.task == 'classification':
            
            preds_xgb = self.xgb.predict_proba(X).argmax(axis=1)
            preds_cat = self.cat.predict_proba(X).argmax(axis=1)
            preds_tf = self.tf.predict_proba(X).argmax(axis=1)
            
            acc_xgb = accuracy_score(preds_xgb, y)
            acc_cat = accuracy_score(preds_cat, y)
            acc_tf = accuracy_score(preds_tf, y)
            
            preds_xgba = self.xgb.predict_proba(X)
            preds_cata = self.cat.predict_proba(X)
            preds_tfa = self.tf.predict_proba(X)

            predsa = (preds_xgba + preds_cata + preds_tfa)/3
            predsa_a=np.argmax(predsa, axis=1)
            acc = accuracy_score(y, predsa_a)

            print('acc_xgb : ', acc_xgb)
            print('acc_cat : ', acc_cat)
            print('acc_tf  : ', acc_tf)
            print('acc/3   : ', acc)    
        
        if self.task == 'regression':
            
            preds_tf = pd.DataFrame(self.tf.predict(X), columns=['preds'])
            preds_xgb = pd.DataFrame(self.xgb.predict(X), columns=['preds'])
            preds_cat = pd.DataFrame(self.cat.predict(X), columns=['preds'])
           
            r2_xgb = r2_score(preds_xgb, y)
            r2_cat = r2_score(preds_cat, y)
            r2_tf = r2_score(preds_tf, y)

            predsa = (preds_xgb + preds_cat + preds_tf)/3
            r2 = r2_score(predsa, y)

            print('r2_xgb : ', r2_xgb)
            print('r2_cat : ', r2_cat)
            print('r2_tf  : ', r2_tf)
            print('r2/3   : ', r2)                
    
    def predict(self, X):
        
        if self.task == 'classification':
            
            preds_xgba = self.xgb.predict_proba(X)
            preds_cata = self.cat.predict_proba(X)
            preds_tfa = self.tf.predict_proba(X)

            predsa = (preds_xgba + preds_cata + preds_tfa)/3
            predsa_a=np.argmax(predsa, axis=1)
            
            return predsa_a
        
        if self.task == 'regression':
            
            preds_tf = pd.DataFrame(self.tf.predict(X), columns=['preds'])
            preds_xgb = pd.DataFrame(self.xgb.predict(X), columns=['preds'])
            preds_cat = pd.DataFrame(self.cat.predict(X), columns=['preds'])
           
            predsa = (preds_xgb + preds_cat + preds_tf)/3
            
            return predsa           
        
    def saveModel(self, name):
        import os
        folder = "trained_models"
        
        current_datetime = datetime.now().strftime("%Y%M%d%H%M%S")
        
        folder_modelu = folder + "/" + name + current_datetime
        
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        if not os.path.exists(folder_modelu):
            os.mkdir(folder_modelu)
        
        
        self.xgb.model.save_model(folder_modelu + "/xgb.json")
        self.cat.model.save_model(folder_modelu + "/cat.json",format="json",export_parameters=None,pool=None)
        self.tf.model.save(folder_modelu + '/tf.h5')
        
        print("Ensamble models were saved to:")
        print(folder_modelu)
        return folder_modelu
        
    def loadModel(self, name, num_class=None):
        
        if self.task == 'classification':
            if num_class != None:
                self.xgbparams = {}
                self.xgbparams['num_class'] = num_class
            
            self.xgb = psxgb.psXGB()
            self.xgb.model = XGBClassifier()
            self.xgb.model.load_model(name + "/xgb.json")
            
            self.cat = pscat.psCAT()
            self.cat.model = CatBoostClassifier()
            self.cat.model.load_model(name + "/cat.json", format="json")
            
        if self.task == 'regression':
            self.xgb = psxgb.psXGB()
            self.xgb.model = XGBRegressor()
            self.xgb.model.load_model(name + "/xgb.json")
            
            self.cat = pscat.psCAT()
            self.cat.model = CatBoostRegressor()
            self.cat.model.load_model(name + "/cat.json", format="json")
        
        self.tf = pstf.psTF()
        self.tf.model = keras.models.load_model(name + '/tf.h5')
        
    def getBestParamsForFit(self, estimator):
        if(estimator == 'xgb'):
            return self.xgbparams
        elif(estimator == 'cat'):
            return self.catparams
        elif(estimator == 'tf'):
            return self.tfparams
        else:
            print('Estimator not recognized')