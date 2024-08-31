import time, random, gc
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, KFold

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Activation, Conv1D, MaxPooling1D, LSTM, Bidirectional, GRU
from keras.callbacks import EarlyStopping
from keras import Sequential, backend
from keras.layers import Dense

import matplotlib.pyplot as plt
import seaborn as sns
palette = ["#365486", "#7FC7D9", "#0F1035", "#7A316F", "#7A316F"]
sns.set_palette(palette=palette)

warnings.filterwarnings("ignore")


class psTF:
    model = None
    params = None
    test_size = 0.2
    randSeed = 123
    eval_metric = "rmse"
    epochs_param = [10000]
    optimizer_param = ["adam"] 
    loss_param = ["mae"]
    batch_size_param = [32]
    early_stopping_rounds_param = [10, 20, 50, 100]
    metric_out = "rmse"
    min_lr = 0.001
    verbose=0
    cv = 5
    task = None
    build_test_size = 0.2
    layers_param = {
        "nn1":{
        1:{"type":"Dense","n":100,"activation":"relu"},
        2:{"type":"Drop","rate":0.2},
        3:{"type":"Dense","n":1,"activation":"relu"},
            }
        }
                   
    def __init__(self, all_params=None):
         if all_params is not None:
            for key, value in all_params.items():
                setattr(self, key, value)
    
    
    
    
    def buildRegressor(self, X , y):
        
        self.task = "regression"
        
        fitted_models = {}
        
        modelCount = 1
            
        # Calculate all possible combinations and time remaining
        allVarsCount = len(self.epochs_param) * len(self.optimizer_param) * len(self.loss_param) * len(self.batch_size_param) * len(self.early_stopping_rounds_param) * len(self.layers_param)
        buildTime = time.time()
        
        print("============== Training started for Tensorflow ===============")
        print("Try to build ", allVarsCount, " models in ", self.cv, " fold(s)")
        print("==============================================================")

        for epochs in self.epochs_param:
            for optimizer in self.optimizer_param:
                for loss in self.loss_param:
                    for batch_size in self.batch_size_param:
                        for early_stopping_rounds in self.early_stopping_rounds_param:
                            for layers in self.layers_param:

                                # Train model
                                ts = time.time()
                                
                                
        
                                cv_models = {}
                                
                                if(self.cv > 1): 
                                
                                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_rounds)
                                    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,min_lr=self.min_lr)
                                
                                    kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.randSeed)
                                    
                                    i = 1
                                    for train_index, test_index in kf.split(X):
                                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                                        XT_train = X_train#np.asarray(X_train).astype(np.float32)
                                        yT_train = y_train#np.asarray(y_train).astype(np.float32)
                                        XT_test = X_test#np.asarray(X_test).astype(np.float32)
                                        yT_test = y_test#np.asarray(y_test).astype(np.float32)

                                        model1 = Sequential()
                                        
                                        li = 1
                                        for l in self.layers_param[layers]:
                                            if self.layers_param[layers][l]["type"] == "Dense":
                                                if li == 1:
                                                    model1.add(Dense(self.layers_param[layers][l]["n"], activation=self.layers_param[layers][l]["activation"], input_shape=(XT_train.shape[1],)))
                                                else:
                                                    model1.add(Dense(self.layers_param[layers][l]["n"], activation=self.layers_param[layers][l]["activation"]))
                                            elif self.layers_param[layers][l]["type"] == "Conv1D":
                                                model1.add(Conv1D(self.layers_param[layers][l]["nf"], self.layers_param[layers][l]["k"], activation=self.layers_param[layers][l]["activation"]))
                                            elif self.layers_param[layers][l]["type"] == "MaxPooling1D":
                                                model1.add(MaxPooling1D(self.layers_param[layers][l]["size"]))
                                            elif self.layers_param[layers][l]["type"] == "Drop":
                                                model1.add(Dropout(self.layers_param[layers][l]["rate"]))
                                            elif self.layers_param[layers][l]["type"] == "Flatten":
                                                model1.add(Flatten())
                                                
                                            li += 1
                                        
                                        model1.compile(optimizer=optimizer, loss=loss, metrics=self.metric_out.split(sep=',', maxsplit=-1))

                                        # fit the model
                                        history = model1.fit(XT_train, yT_train, epochs=epochs, batch_size=batch_size, verbose=self.verbose, validation_data=(XT_test, yT_test), callbacks=[es,lr])
                                        
                                        tt = time.time() - ts

                                        preds = model1.predict(XT_test)
                                        
                                        rmse = mean_squared_error(yT_test, preds, squared=False)
                                        mae = mean_absolute_error(yT_test, preds)
                                        r2 = r2_score(yT_test, preds)
                                        mape = mean_absolute_percentage_error(yT_test, preds)
                                        medae = median_absolute_error(yT_test, preds)
                                        
                                        cv_models[i] = {"rmse":rmse, "mae":mae, "mape": mape, "medae": medae,"r2":r2,  "model": model1}
                                        i += 1
                                        
                                elif(self.cv == 1):
                                
                                    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=early_stopping_rounds)
                                    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5,patience=10,min_lr=self.min_lr)
                                    
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.build_test_size, random_state=self.randSeed)
                                    
                                    model1 = Sequential()
                                    
                                    li = 1
                                    for l in self.layers_param[layers]:
                                        if self.layers_param[layers][l]["type"] == "Dense":
                                            if li == 1:
                                                model1.add(Dense(self.layers_param[layers][l]["n"], activation=self.layers_param[layers][l]["activation"], input_shape=(X_train.shape[1],)))
                                            else:
                                                model1.add(Dense(self.layers_param[layers][l]["n"], activation=self.layers_param[layers][l]["activation"]))
                                        elif self.layers_param[layers][l]["type"] == "Conv1D":
                                            model1.add(Conv1D(self.layers_param[layers][l]["nf"], self.layers_param[layers][l]["k"], activation=self.layers_param[layers][l]["activation"]))
                                        elif self.layers_param[layers][l]["type"] == "MaxPooling1D":
                                            model1.add(MaxPooling1D(self.layers_param[layers][l]["size"]))
                                        elif self.layers_param[layers][l]["type"] == "Drop":
                                            model1.add(Dropout(self.layers_param[layers][l]["rate"]))
                                        elif self.layers_param[layers][l]["type"] == "Flatten":
                                            model1.add(Flatten())
                                            
                                        li += 1
                                    
                                    model1.compile(optimizer=optimizer, loss=loss, metrics=self.metric_out.split(sep=',', maxsplit=-1))

                                    # fit the model
                                    history = model1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=self.verbose, validation_data=(X_test, y_test), callbacks=[es,lr])
                                    
                                    tt = time.time() - ts

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
                                                                "epochs":epochs,
                                                                "optimizer":optimizer,
                                                                "loss":loss,
                                                                "batch_size":batch_size,
                                                                "early_stopping_rounds":early_stopping_rounds,
                                                                "layers":self.layers_param[layers],
                                                                "min_lr": self.min_lr,
                                                                "time":tt,
                                                                "history": history,
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
        
        backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

    def buildClassifier(self, X , y):
        
        self.task = "classification"
        
        fitted_models = {}
        
        modelCount = 1
            
        # Calculate all possible combinations and time remaining
        allVarsCount = len(self.epochs_param) * len(self.optimizer_param) * len(self.loss_param) * len(self.batch_size_param) * len(self.early_stopping_rounds_param) * len(self.layers_param)
        buildTime = time.time()
        
        print("=========== Training started for Tensorflow ============")
        print("Try to build ", allVarsCount, " models in ", self.cv, " fold(s)")
        print("========================================================")

        for epochs in self.epochs_param:
            for optimizer in self.optimizer_param:
                for loss in self.loss_param:
                    for batch_size in self.batch_size_param:
                        for early_stopping_rounds in self.early_stopping_rounds_param:
                            for layers in self.layers_param:

                                # Train model
                                ts = time.time()
                                
                                cv_models = {}
                                                 
                                if(self.cv > 1):                                                                
                                    
                                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_rounds)
                                    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,min_lr=self.min_lr)
                                    
                                    kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.randSeed)
                                                                    
                                    i = 1
                                    for train_index, test_index in kf.split(X):
                                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                                        
                                        XT_train = X_train#np.asarray(X_train).astype(np.float32)
                                        yT_train = y_train#np.asarray(y_train).astype(np.float32)
                                        XT_test = X_test#np.asarray(X_test).astype(np.float32)
                                        yT_test = y_test#np.asarray(y_test).astype(np.float32)
                                        
                                        y_test_a=np.argmax(y_test, axis=1)

                                        model1 = Sequential()
                                        
                                        li = 1
                                        for l in self.layers_param[layers]:
                                            if self.layers_param[layers][l]["type"] == "Dense":
                                                if li == 1:
                                                    model1.add(Dense(self.layers_param[layers][l]["n"], activation=self.layers_param[layers][l]["activation"], input_shape=(XT_train.shape[1],)))
                                                else:
                                                    model1.add(Dense(self.layers_param[layers][l]["n"], activation=self.layers_param[layers][l]["activation"]))
                                            elif self.layers_param[layers][l]["type"] == "Conv1D":
                                                model1.add(Conv1D(self.layers_param[layers][l]["nf"], self.layers_param[layers][l]["k"], activation=self.layers_param[layers][l]["activation"]))
                                            elif self.layers_param[layers][l]["type"] == "MaxPooling1D":
                                                model1.add(MaxPooling1D(self.layers_param[layers][l]["size"]))
                                            elif self.layers_param[layers][l]["type"] == "Drop":
                                                model1.add(Dropout(self.layers_param[layers][l]["rate"]))
                                            elif self.layers_param[layers][l]["type"] == "Flatten":
                                                model1.add(Flatten())
                                            
                                            li += 1
                                        
                                        model1.compile(optimizer=optimizer, loss=loss, metrics=self.metric_out.split(sep=',', maxsplit=-1)) #, 'Precision', 'Recall'
                                        history = model1.fit(XT_train, yT_train, epochs=epochs, batch_size=batch_size, verbose=self.verbose, validation_data=(XT_test, yT_test),callbacks=[es,lr])#callbacks=[es,lr]
                                        
                                        preds = model1.predict(XT_test)
                                        y_pred_a=np.argmax(preds, axis=1)
                                        
                                        accuracy = accuracy_score(y_pred_a, yT_test)
                                        precision = precision_score(y_pred_a, yT_test, average='weighted')
                                        recall = recall_score(y_pred_a, yT_test, average='weighted')
                                        f1 = f1_score(y_pred_a, yT_test, average='weighted')
                                        
                                        cv_models[i] = {"accuracy":accuracy, "precision":precision, "recall": recall, "f1":f1, "model": model1}
                                        i += 1
                                elif(self.cv == 1):
                                    
                                    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=early_stopping_rounds)
                                    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.1,patience=5,min_lr=self.min_lr)
                                    
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.build_test_size, random_state=self.randSeed)

                                    model1 = Sequential()
                                    
                                    li = 1
                                    for l in self.layers_param[layers]:
                                        if self.layers_param[layers][l]["type"] == "Dense":
                                            if li == 1:
                                                model1.add(Dense(self.layers_param[layers][l]["n"], activation=self.layers_param[layers][l]["activation"], input_shape=(X_train.shape[1],)))
                                            else:
                                                model1.add(Dense(self.layers_param[layers][l]["n"], activation=self.layers_param[layers][l]["activation"]))
                                        elif self.layers_param[layers][l]["type"] == "Conv1D":
                                            model1.add(Conv1D(self.layers_param[layers][l]["nf"], self.layers_param[layers][l]["k"], activation=self.layers_param[layers][l]["activation"]))
                                        elif self.layers_param[layers][l]["type"] == "MaxPooling1D":
                                            model1.add(MaxPooling1D(self.layers_param[layers][l]["size"]))
                                        elif self.layers_param[layers][l]["type"] == "Drop":
                                            model1.add(Dropout(self.layers_param[layers][l]["rate"]))
                                        elif self.layers_param[layers][l]["type"] == "Flatten":
                                            model1.add(Flatten())
                                        
                                        li += 1
                                    
                                    model1.compile(optimizer=optimizer, loss=loss, metrics=self.metric_out.split(sep=',', maxsplit=-1)) #, 'Precision', 'Recall'
                                    history = model1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=self.verbose, validation_data=(X_test, y_test),callbacks=[es,lr])#callbacks=[es,lr]
                                    
                                    preds = model1.predict(X_test)
                                    y_pred_a=np.argmax(preds, axis=1)
                                    
                                    accuracy = accuracy_score(y_pred_a, y_test)
                                    precision = precision_score(y_pred_a, y_test, average='weighted')
                                    recall = recall_score(y_pred_a, y_test, average='weighted')
                                    f1 = f1_score(y_pred_a, y_test, average='weighted')
                                    
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
                                                                "epochs":epochs,
                                                                "optimizer":optimizer,
                                                                "loss":loss,
                                                                "batch_size":batch_size,
                                                                "early_stopping_rounds":early_stopping_rounds,
                                                                "layers":self.layers_param[layers],
                                                                "min_lr": self.min_lr,
                                                                "time":tt,
                                                                "history": history,
                                                                "model": model1
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

        XT = np.asarray(X).astype(np.float32)
        yT = np.asarray(y).astype(np.float32)
        
        preds = best_model['model'].values[0].predict(XT)
        y_pred_a=np.argmax(preds, axis=1)
        yTa = np.argmax(yT, axis=1)
        print("=========== Report for best model in Tensorflow ============")
        print(classification_report(yT, y_pred_a))
        
        cm = confusion_matrix(yT, y_pred_a, normalize='true')
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        disp.ax_.set_title("Confusion Matrix for Tensorflow")
    
        self.model = best_model['model'].values[0]
        self.params = best_model
        
        backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()


    def predict_proba(self,X):
        Xn = np.asarray(X).astype(np.float32)
        preds_tf = self.model.predict(Xn)
        
        return preds_tf
    
    def predict(self,X):
        Xn = np.asarray(X).astype(np.float32)
        preds_tf = self.model.predict(Xn)
        
        if self.task == 'classification': 
            return np.argmax(preds_tf, axis=1) 
        else:
            return preds_tf