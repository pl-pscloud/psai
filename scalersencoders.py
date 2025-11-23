from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer, FunctionTransformer, LabelEncoder, OrdinalEncoder, MinMaxScaler, TargetEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import RFE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
#from category_encoders import TargetEncoder
import numpy as np

# Define the identity function for feature names
def identity_feature_names_out(feature_names_in):
    return feature_names_in

# Create the transformer
Logtransformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=None,
    validate=True,
    feature_names_out=identity_feature_names_out
)

dim_config = {
    'none': None,
    'pca': PCA(n_components=5,svd_solver='auto'), 
    'pca_10': PCA(n_components=10,svd_solver='auto'),
    'kpca': KernelPCA(n_components=5, kernel='rbf', gamma=1),
    'kpca_10': KernelPCA(n_components=10, kernel='rbf', gamma=0.1),
    'svd': TruncatedSVD(n_components=5),
    'svd_10': TruncatedSVD(n_components=10),
}

numerical_imputer_config = {
    'none': None,
    'mean': SimpleImputer(strategy='mean'),
    'median': SimpleImputer(strategy='median'),
    'constant': SimpleImputer(strategy='constant', fill_value=-1),
    'knn': KNNImputer(n_neighbors=5),
    'knn_10': KNNImputer(n_neighbors=10),
    'iterative': IterativeImputer(),
}

numerical_scaler_config = {
    'none': None,
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(feature_range=(0, 1)),
    'robust': RobustScaler(),
    'quantile': QuantileTransformer(output_distribution='normal'),
    'yeo-johnson': PowerTransformer(method='yeo-johnson', standardize=True),
    'box-cox': PowerTransformer(method='box-cox', standardize=True),
    'log': Logtransformer,
}

cat_imputer_config = {
    'none': None,
    'most_frequent': SimpleImputer(strategy='most_frequent'),
    'constant': SimpleImputer(strategy='constant', fill_value='Unknown')
}

cat_encoder_config = {
    'none': None,
    'onehot': OneHotEncoder(handle_unknown='ignore'),
    'target': TargetEncoder(smooth=0.1),
    'target_0.5': TargetEncoder(smooth=0.5),
    'target_1': TargetEncoder(smooth=1),
    'target_10': TargetEncoder(smooth=10),
    'hashing': FeatureHasher(n_features=10, input_type='string'),
    'label': LabelEncoder(),
    'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
}

cat_scaler_config = {
    'none': None,
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(feature_range=(0, 1)),
    'robust': RobustScaler(),
    'quantile': QuantileTransformer(n_quantiles=10, random_state=0),
    'log': FunctionTransformer(np.log1p, validate=True),
}

def create_preprocessor(config, X):

    columns = categorize_features(X)
    # Create numerical transformer with hyperparameterized SimpleImputer
    numerical_transformer = Pipeline(steps=[
        ('imputer', numerical_imputer_config[config['numerical']['imputer']]),
        ('scaler', numerical_scaler_config[config['numerical']['scaler']]),
    ])

    # Create skew transformer
    skew_transformer = Pipeline(steps=[
        ('imputer', numerical_imputer_config[config['skewed']['imputer']]),
        ('scaler', numerical_scaler_config[config['skewed']['scaler']]),
    ])


    # Create outlier transformer
    outlier_transformer = Pipeline(steps=[
        ('imputer', numerical_imputer_config[config['outlier']['imputer']]),
        ('scaler', numerical_scaler_config[config['outlier']['scaler']]),
    ])

    # Create low cardinality transformer 
    low_cardinality_transformer = Pipeline(steps=[
        ('imputer', cat_imputer_config[config['low_cardinality']['imputer']]),
        ('encoder', cat_encoder_config[config['low_cardinality']['encoder']]),
        ('scaler', cat_scaler_config[config['low_cardinality']['scaler']]),
    ])

    # Create high cardinality
    high_cardinality_transformer = Pipeline(steps=[
        ('imputer', cat_imputer_config[config['high_cardinality']['imputer']]),
        ('encoder', cat_encoder_config[config['high_cardinality']['encoder']]),
        ('scaler', cat_scaler_config[config['high_cardinality']['scaler']]),
    ])

    # Combine all pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('high', high_cardinality_transformer, columns['high_cols']),  # High-cardinality categorical columns
            ('low', low_cardinality_transformer, columns['low_cols']),  # Low-cardinality categorical columns
            ('num', numerical_transformer, columns['nskewed_cols']),  # Numerical columns
            ('skew', skew_transformer, columns['skewed_cols']),  # Numerical skew columns
            ('outlier', outlier_transformer, columns['outlier_cols']),  # Numerical skew columns 
        ]
    )

    # Add dimension reduction if method is specified and not 'none'
    dim_reduction_config = config.get('dimension_reduction', {})
    method = dim_reduction_config.get('method', 'none')
    
    if method != 'none':
        dim_transformer = dim_config.get(method)
        
        if dim_transformer is not None:
             preprocessor = Pipeline(steps=[
                 ('preprocessor', preprocessor),
                 ('dim_reduction', dim_transformer)
             ])

    return preprocessor, columns

def categorize_features(X):
    # Separate categorical and numerical features
    cat_cols = X.select_dtypes(include=['object','category']).columns
    num_cols = X.select_dtypes(include=['int32', 'int64','float64']).columns

    # Determine cardinality for categorical features
    cardinality = X[cat_cols].nunique()
    threshold = 10
    high_cols = cardinality[cardinality > threshold].index.tolist()
    low_cols = cardinality[cardinality <= threshold].index.tolist()

    # Determine skewness for numerical features
    skew_threshold = 0.5
    skewness = X[num_cols].apply(lambda x: x.skew())
    skewed_cols = skewness[skewness.abs() > skew_threshold].index.tolist()  # Threshold for skewness
    nskewed_cols = skewness[skewness.abs() <= skew_threshold].index.tolist()

    outlier_cols = []
    for col in num_cols:
        Q1 = np.percentile(X[col], 25)
        Q3 = np.percentile(X[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (X[col] < lower_bound) | (X[col] > upper_bound)
        outlier_ratio = outliers.mean()
        # Lege ein Threshold fest, z.B. wenn mehr als 5% Ausreißer
        if outlier_ratio > 0.05:
            outlier_cols.append(col)

    skewed_cols = [col for col in skewed_cols if col not in outlier_cols]
    nskewed_cols = [col for col in nskewed_cols if col not in outlier_cols]

    # Prepare embedding_info for high cardinality columns
    embedding_info = {}

    # Set a rule for choosing embedding dimensions based on cardinality
    for col in high_cols:
        number_of_categories = X[col].nunique()  # Calculate the number of unique categories
        embedding_dim = min(2 + (number_of_categories // 8), 50)  # Decide on the embedding dimension
        embedding_info[col] = (number_of_categories, embedding_dim)

    print(f'Num cols: {nskewed_cols}')
    print(f'Num skew cols: {skewed_cols}')
    print(f'Outliers cols: {outlier_cols}')
    print(f'Cat cols: {low_cols}')
    print(f'Cat high cardinality cols: {high_cols}')
    print(f'Embeddings: {embedding_info}')

    return {
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'low_cols': low_cols,
        'high_cols': high_cols,
        'skewed_cols': skewed_cols,
        'nskewed_cols': nskewed_cols,
        'outlier_cols': outlier_cols,
        'embedding_info': embedding_info
    }