import keras.layers
from keras import Sequential
from keras.src.layers import Bidirectional, LSTM, Dense, Dropout, Flatten
from keras.src.optimizers import Adam




def getModel(shape,dropout=0):
    #fornito shape di dataset suddiviso in timestep, ritorna modello
    #Se si vuole si può fornire tasso didropout dell'ultimo livello della parte LSTM del modello

    my_model = Sequential()

    #Parte LSTM
    my_model.add(Bidirectional(LSTM(units=39*8, return_sequences=True,input_shape=(shape[1], shape[2]))))
    my_model.add(Bidirectional(LSTM(units=39*6, return_sequences=True)))
    my_model.add(Bidirectional(LSTM(units=39*8, return_sequences=True)))
    my_model.add(Bidirectional(LSTM(units=39*4, return_sequences=True)))
    my_model.add(Bidirectional(LSTM(units=39*8, return_sequences=True,dropout=0.005)))





    #Parte CNN
    my_model.add(keras.layers.Conv1D(kernel_size=3,filters=64,activation="relu"))
    my_model.add(Dense(units=64,activation="relu"))
    my_model.add(keras.layers.MaxPooling1D(pool_size=2))
    my_model.add(keras.layers.Conv1D(kernel_size=2,filters=128,activation="relu"))
    my_model.add(keras.layers.Flatten())
    my_model.add(Dense(units=32,activation="relu"))


    my_model.add(Dense(units=1))
    custom_optimizer = Adam(learning_rate=0.00001)
    #my_model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
    #my_model.compile(optimizer=custom_optimizer, loss='mean_squared_error',
    #                 metrics=['mape', 'mae', 'mse'])

    my_model.compile(optimizer=custom_optimizer, loss='mean_absolute_error',
                     metrics=['mape', 'mae', 'mse'])

    return my_model
