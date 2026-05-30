import keras.layers
from keras.src.layers import Bidirectional, LSTM, Dense, Dropout, Flatten
from keras.src.optimizers import Adam



def getModel(shape,dropout=0):
    #fornito shape di dataset suddiviso in timestep, ritorna modello
    #Se si vuole si può fornire tasso didropout dell'ultimo livello della parte LSTM del modello


    #Parte CNN
    inputs=keras.Input(shape=(shape[1],shape[2]))

    conv1D=keras.layers.Conv1D(kernel_size=3,filters=64,activation="relu")
    outCNN=conv1D(inputs)
    dense1= Dense(units=64,activation="relu")
    outCNN=dense1(outCNN)
    maxPooling=keras.layers.MaxPooling1D(pool_size=2)
    outCNN=maxPooling(outCNN)
    conv1D_2=keras.layers.Conv1D(kernel_size=2,filters=128,activation="relu")
    outCNN=(conv1D_2(outCNN))
    flatten= keras.layers.Flatten()
    outCNN=flatten(outCNN)
    dense2= Dense(units=32,activation="relu")
    outCNN=dense2(outCNN)

    denseFinalCNN =Dense(units=1)
    outCNN=denseFinalCNN(outCNN)



    #Parte LSTM
    bidir1 = Bidirectional(LSTM(units=39*8, return_sequences=True))
    outLSTM = bidir1(inputs)
    bidir2=Bidirectional(LSTM(units=39*6, return_sequences=True))
    outLSTM = bidir2(outLSTM)
    bidir3=Bidirectional(LSTM(units=39*8, return_sequences=True))
    outLSTM = bidir3(outLSTM)
    bidir4=Bidirectional(LSTM(units=39*4, return_sequences=True))
    outLSTM = bidir4(outLSTM)
    bidir5=Bidirectional(LSTM(units=39*8, return_sequences=True,dropout=dropout))
    outLSTM = bidir5(outLSTM)
    flatten6=Flatten()
    outLSTM=flatten6(outLSTM)
    denseFinalLSTM =Dense(units=1)
    outLSTM = denseFinalLSTM(outLSTM)

    #output finale
    outputs=keras.layers.Average()([outCNN,outLSTM])

    #modello
    my_model=keras.Model(inputs=inputs,outputs=outputs,name="ParallelModel")
    custom_optimizer = Adam(learning_rate=0.00001)
    #my_model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
    #my_model.compile(optimizer=custom_optimizer, loss='mean_squared_error',
    #                 metrics=['mape', 'mae', 'mse'])

    my_model.compile(optimizer=custom_optimizer, loss='mean_absolute_error',
                     metrics=['mape', 'mae', 'mse'])




    return my_model