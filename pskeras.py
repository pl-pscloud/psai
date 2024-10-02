# Define the Keras model with input and output shapes
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

def create_keras_model(meta, net={1:['dense',1024,'relu'],2:['dropout',0.1],3:['dense',256,'relu']}, optimizer='adam'):
    model = Sequential()
    model.add(Input(shape=(meta["n_features_in_"],)))
    
    for k,v in net.items():
        if v[0] == 'dense':
            model.add(Dense(v[1], activation=v[2]))
        if v[0] == 'dropout':
            model.add(Dropout(rate=v[1]))

    model.add(Dense(meta["n_outputs_"]))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Pipeline for KerasRegressor using SciKeras
keras_regressor = KerasRegressor(
    model=create_keras_model,
    epochs=100,
    batch_size=32,
    verbose=0,
    optimizer='adam',
    model__net={1:['dense',1024,'relu'],2:['dropout',0.1],3:['dense',256,'relu']},
    callbacks=[EarlyStopping(monitor='loss', patience=10, mode='min', restore_best_weights=True)],
)