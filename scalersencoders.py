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

#imputers, encoders etc
dim_config = {
    0: None,
    1: PCA(n_components=5,svd_solver='auto'), 
    2: PCA(n_components=10,svd_solver='auto'),
    3: KernelPCA(n_components=5, kernel='rbf', gamma=1),
    4: KernelPCA(n_components=10, kernel='rbf', gamma=0.1),
    5: TruncatedSVD(n_components=5),
    6: TruncatedSVD(n_components=10),
    
}

numerical_imputer_config = {
    0: None,
    1: SimpleImputer(strategy='mean'),
    2: SimpleImputer(strategy='median'),
    3: SimpleImputer(strategy='constant', fill_value='-1'),
    4: KNNImputer(n_neighbors=5),
    5: KNNImputer(n_neighbors=10),
    6: IterativeImputer(),
}

numerical_scaler_config = {
    0: None,
    1: StandardScaler(),
    2: MinMaxScaler(feature_range=(0, 1)),
    3: RobustScaler(),
    4: QuantileTransformer(output_distribution='normal'),
    5: PowerTransformer(method='yeo-johnson', standardize=True),
    6: PowerTransformer(method='box-cox', standardize=True),
    7: Logtransformer,
}

cat_imputer_config = {
    0: None,
    1: SimpleImputer(strategy='most_frequent'),
    2: SimpleImputer(strategy='constant', fill_value='Unknown')
}

cat_encoder_config = {
    0: None,
    1: OneHotEncoder(handle_unknown='ignore'),
    2: TargetEncoder(smooth=0.1),
    3: TargetEncoder(smooth=0.5),
    4: TargetEncoder(smooth=1),
    5: TargetEncoder(smooth=10),
    6: FeatureHasher(n_features=10, input_type='string'),
    7: LabelEncoder(),
    8: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
}


cat_scaler_config = {
    0: None,
    1: StandardScaler(),
    2: MinMaxScaler(feature_range=(0, 1)),
    3: RobustScaler(),
    4: QuantileTransformer(n_quantiles=10, random_state=0),
    5: FunctionTransformer(np.log1p, validate=True),
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