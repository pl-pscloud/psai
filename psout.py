import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder
from feature_engine.selection import DropCorrelatedFeatures
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns



def isoForest(df, col, contamination=0.1):
    
    dfC = df.copy()

    isoF = IsolationForest(contamination=contamination)
    isoF.fit(dfC[col])

    dfC['anomally']=isoF.predict(dfC[col])
    anomally = dfC[dfC['anomally'] == -1]
    dfC = dfC[dfC['anomally'] == 1]
    
    dfC.drop('anomally', axis='columns', inplace=True)
    anomally.drop('anomally', axis='columns', inplace=True)
    
    return dfC, anomally

def scalerStd(df, col = None):
    sc = StandardScaler()

    if col is None:
        X1 = sc.fit_transform(df)
        X1df = pd.DataFrame(X1, index=df.index)
        
        return X1df, sc
    else:
        X1 = sc.fit_transform(df[col])
        X1df = pd.DataFrame(X1, columns=df[col].columns, index=df.index)

        dfC = df.copy()
        dfC[col] = X1df[col]
        
        return dfC, sc

def scalerNorm(df, col = None):
    sc = MinMaxScaler(feature_range=(0, 1))
    
    if col is None:
        X1 = sc.fit_transform(df)
        X1df = pd.DataFrame(X1, index=df.index)
        
        return X1df, sc
    else:
        X1 = sc.fit_transform(df[col])
        X1df = pd.DataFrame(X1, columns=df[col].columns, index=df.index)

        dfC = df.copy()
        dfC[col] = X1df[col]
        
        return dfC, sc

def oneHot(df, col = None):
    
    dfC = df.copy()
    if col == None:
        dfC = pd.get_dummies(dfC)
    else:
        dfC = pd.get_dummies(dfC, columns=col)
    
    return dfC

def labelEnc(df):
    
    dfC = df.copy()
    enc = LabelEncoder()
    dfC = enc.fit_transform(dfC)
    
    return dfC,enc


def dropCorr(df, threshold=0.85):
    dfC = df.copy()
    corrolated = DropCorrelatedFeatures(method="pearson", threshold=0.85)
    corrolated.fit(dfC)  
    dfC = corrolated.transform(dfC)
    return dfC

def dropRowsZero(df, cols):
    dfC = df.copy()
    for col in cols:
        dfC = dfC[dfC[col] != 0]
    return dfC

def smote(X,y):
    oversample = SMOTE()
    return oversample.fit_resample(X, y)

def nullInDataset(df):
    null_info = pd.DataFrame({
    'Null count': df.isnull().sum(),
    'Null percent': (df.isnull().sum() / len(df)) * 100
    })
    
    null_info = null_info.sort_values(by='Null percent', ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=null_info.index, y=null_info['Null percent'])
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.ylabel('Null [%]', fontsize=9)
    plt.xlabel('Data columns', fontsize=9)
    plt.title('Null percent - sorted', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    return null_info

def removeColumnsHighNulls(df, threshold_percent):
    null_counts = df.isnull().sum()
    percentage_nulls = (null_counts / len(df)) * 100

    columns_to_remove = percentage_nulls[percentage_nulls > threshold_percent].index

    df_cleaned = df.drop(columns=columns_to_remove)

    return df_cleaned

def outliers(df, cols, remove = False):
    for col in cols:
        # Wykres pudełkowy dla zmiennej BMI
        plt.figure(figsize=(8,5))
        sns.boxplot(x=df[col])
        plt.title(col)
        plt.xlabel(col)
        plt.show()

        # Obliczenie IQR dla BMI
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Określenie granic dla wartości odstających
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Zidentyfikowanie wartości odstających
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f'Count of outliers: {outliers.shape[0]}')
        
        if remove == True:
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df
        
