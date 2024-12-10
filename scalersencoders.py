from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer, FunctionTransformer, LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import RFE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from category_encoders import TargetEncoder
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
    3: SimpleImputer(strategy='constant'),
    4: KNNImputer(n_neighbors=10),
    5: IterativeImputer(),
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

skew_imputer_config = {
    0: None,
    1: SimpleImputer(strategy='mean'), 
    2: SimpleImputer(strategy='median'),
    3: SimpleImputer(strategy='constant'), 
    4: KNNImputer(n_neighbors=10), 
    5: IterativeImputer(),
}

skew_scaler_config = {
    0: None,
    1: StandardScaler(),    
    2: MinMaxScaler(feature_range=(0, 1)),
    3: RobustScaler(),
    4: QuantileTransformer(output_distribution='normal'),
    5: PowerTransformer(method='yeo-johnson', standardize=True),
    6: PowerTransformer(method='box-cox', standardize=True),
    7: Logtransformer,
}

high_imputer_config = {
    0: None,
    1: SimpleImputer(strategy='most_frequent'),
}

low_imputer_config = {
    0: None,
    1: SimpleImputer(strategy='most_frequent'),
}

low_encoder_config = {
    0: None,
    1: OneHotEncoder(handle_unknown='ignore'),
    2: TargetEncoder(smoothing=1),
    3: TargetEncoder(smoothing=100),
    4: FeatureHasher(n_features=10, input_type='string'),
    5: LabelEncoder(),
    6: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
}

high_encoder_config = {
    0: None,
    1: OneHotEncoder(handle_unknown='ignore'),
    2: TargetEncoder(smoothing=1),
    3: TargetEncoder(smoothing=100),
    4: FeatureHasher(n_features=10, input_type='string'),
    5: LabelEncoder(),
    6: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
}

low_scaler_config = {
    0: None,
    1: StandardScaler(), 
    2: RobustScaler(),
    3: QuantileTransformer(n_quantiles=10, random_state=0),
    4: FunctionTransformer(np.log1p, validate=True),
}

high_scaler_config = {
    0: None,
    1: StandardScaler(), 
    2: RobustScaler(),
    3: QuantileTransformer(n_quantiles=10, random_state=0),
}
