import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
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
    dfC = df.copy()

    if col is None:
        numcols = dfC.select_dtypes(include=['int64', 'float64']).columns.to_numpy()
        
        X1 = sc.fit_transform(dfC[numcols])
        X1df = pd.DataFrame(X1, columns=dfC[numcols].columns, index=dfC.index)
        
        dfC[numcols] = X1df[numcols]
        
        return dfC, sc, numcols
    else:
        X1 = sc.fit_transform(dfC[col])
        X1df = pd.DataFrame(X1, columns=df[col].columns, index=dfC.index)

        dfC[col] = X1df[col]
        
        return dfC, sc, col

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

def oneHot(df, catcols = None):
    
    dfC = df.copy()
    
    if catcols is None:
        catcols = dfC.select_dtypes(include=['object']).columns
    
    enc = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    dfE = pd.DataFrame(enc.fit_transform(dfC[catcols]))
    dfE.index = dfC.index
    
    return dfE, enc, catcols

def labelEnc(df,col):
    
    dfC = df.copy()
    enc = LabelEncoder()
    dfC[col] = enc.fit_transform(dfC[col])
    
    return dfC,enc


def dropCorr(df, threshold=0.85):
    dfC = df.copy()
    corrolated = DropCorrelatedFeatures(method="pearson", threshold=0.85)
    corrolated.fit(dfC)  
    dfC = corrolated.transform(dfC)
    
    return dfC

def simpleImputer(df, cat_strategy = 'most_frequent', num_strategy = 'constant', cols = None):
    dfC = df.copy()
    numerical_imputer = SimpleImputer(strategy=num_strategy)
    categorical_imputer = SimpleImputer(strategy=cat_strategy)
    numcols = None
    catcols = None
    
    if cols ==None:
        numcols = dfC.select_dtypes(include=['int64', 'float64']).columns
        catcols = dfC.select_dtypes(include=['object']).columns
        
        dfC[numcols] = numerical_imputer.fit_transform(dfC[numcols])
        dfC[catcols] = categorical_imputer.fit_transform(dfC[catcols])
        
    else:
        for col in cols:
            if dfC[col].dtype in ['int64','float64']:
                dfC[col] = numerical_imputer.fit_transform(dfC[col])
            
            if dfC[col].dtype in ['object']:
                dfC[col] = categorical_imputer.fit_transform(dfC[col])
    
    return dfC, categorical_imputer, numerical_imputer, catcols.to_numpy(), numcols.to_numpy()

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
    
    dfC = df.copy()
    
    for col in cols:
        # Wykres pudełkowy dla zmiennej BMI
        plt.figure(figsize=(8,5))
        sns.boxplot(x=df[col])
        plt.title(col)
        plt.xlabel(col)
        plt.show()

        # Obliczenie IQR dla BMI
        Q1 = dfC[col].quantile(0.25)
        Q3 = dfC[col].quantile(0.75)
        IQR = Q3 - Q1

        # Określenie granic dla wartości odstających
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Zidentyfikowanie wartości odstających
        outliers = dfC[(dfC[col] < lower_bound) | (dfC[col] > upper_bound)]
        print(f'Count of outliers: {outliers.shape[0]}')
        
        if remove == True:
            dfC = dfC[(dfC[col] >= lower_bound) & (dfC[col] <= upper_bound)]
        
    return dfC
        
def removeColumnsWithHighCardinality(df, threshold = 10):
    high_cardinality_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() > threshold]
    
    dfC = df.copy()
    dfC = dfC.drop(columns=high_cardinality_cols)
    
    return dfC, high_cardinality_cols
    