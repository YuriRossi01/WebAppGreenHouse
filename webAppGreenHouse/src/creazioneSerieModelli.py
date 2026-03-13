import os
from sklearn.metrics import (mean_absolute_error as mae)
from sklearn.metrics import (mean_absolute_percentage_error as mape)
from gestioneDati import *
import matplotlib.pyplot as plt

from src.standardModelli import model, modelCNN_LSTM, modelLSTM_CNN, modelCLPARALLEL

#FILE_MODELLO_KERAS = './dati/keras_st4_model.h5'
FILE_DATI_CSV = './dati/Dataset_sens_'
FILE_TEST_CSV = './dati/Test_sens_4.csv'
FILE_MODELLO_KERAS ='./dati/my_model.h5'
dirModelli='./modelli'
# time_lag = 7
epochs = 300

time_lag = 7
nunits = 256
thisdropout = 0.005
batch_size = 70
validation_split = 0.2

def main():

   tipo=["LSTM","CNN_LSTM","LSTM_CNN","CNN_LSTM_PAR"]
   for i in range(1,9):
       creaModello(tipo[3],i,True)




def creaModello(tipo,sensore,shuffle):
    #Funzione copia incolla e rivisitazione leggera del file creazioneModelloManual.py
    #Funzionalità identica, ma senza input da parte di utente
    #Dato il tipo di modello ( string valori specifici), il sensore (int) e shuffle (true/false)
    #Crea modello del tipo indicato, usando i dati del sensore indicato rimescolati o ordinati (shuffle)



    dataDescr = 'base'
    fileDati=FILE_DATI_CSV+str(sensore)+'.csv'
    (trainX, trainY, testX, testY) = getTrainXYTestXY(fileDati, dataDescr, time_lag, shuffle)
    testY=testY[:testY.size-(time_lag-1)]
    nomePercorso=dirModelli
    if(tipo=='LSTM'):
        my_model = model.getModel(trainX.shape,dropout=thisdropout)
        nomePercorso=nomePercorso+'/modelliLSTM/'
    elif(tipo=='CNN_LSTM'):
        my_model=modelCNN_LSTM.getModel(trainX.shape,dropout=thisdropout)
        nomePercorso=nomePercorso+'/modelliCNNLSTM/'
    elif(tipo=='LSTM_CNN'):
        my_model=modelLSTM_CNN.getModel(trainX.shape,dropout=thisdropout)
        nomePercorso=nomePercorso+'/modelliLSTMCNN/'
    elif(tipo=='CNN_LSTM_PAR'):
        my_model=modelCLPARALLEL.getModel(trainX.shape,dropout=thisdropout)
        nomePercorso=nomePercorso+'/modelliCNNLSTMPAR/'
    if(not os.path.exists(nomePercorso)):
        if(tipo=='LSTM'):
            os.mkdir("./modelli/modelliLSTM")
        elif(tipo=="CNN_LSTM"):
            os.mkdir("./modelli/modelliCNNLSTM")
        elif(tipo=="LSTM_CNN"):
            os.mkdir("./modelli/modelliLSTMCNN")
        elif(tipo=="CNN_LSTM_PAR"):
            os.mkdir("./modelli/modelliCNNLSTMPAR")


    nomeCartella = nomePercorso+tipo+'_model_'+str(sensore)
    if(shuffle):
        nomeCartella=nomeCartella+"_shuffle"


    i=0
    if(os.path.exists(nomeCartella)):
        print("errore, modello dello stesso \"tipo\" esiste già, verrà creata cartella duplicata.")
        errore=True
        while(errore):
            nomeCartellaTemp=nomeCartella+"_"+str(i)
            if(os.path.exists(nomeCartellaTemp)):
                i=i+1
            else:
                nomeCartella=nomeCartellaTemp
                errore =False

    os.mkdir(nomeCartella)
    nomeCartella=nomeCartella+"/"
    nomeFileModello=nomeCartella+"model.keras"
    history =my_model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,
                          validation_split=validation_split, verbose=1)
    my_model.save(nomeFileModello)

    plt=creaGrafico(history,"mape",epochs)
    plt.savefig(nomeCartella+"mapeGraph.png")
    plt.clf()

    plt=creaGrafico(history,"mae",epochs)
    plt.savefig(nomeCartella+"maeGraph.png")
    plt.clf()


    result = my_model.predict(testX, batch_size=1)




    print("modello "+str(i)+" di tipo:"+tipo+" fatto-----------------------------------------------------------------------")

    nomeFileData=nomeCartella+"data.txt"
    f = open(nomeFileData, "a")
    text="Ultimi valori per ogni key:\n"
    for key in history.history.keys():
        text=text+"-"+key+": "+str(history.history[key][epochs-1])
        text=text+"\n"

    text=text+"\nValori rispetto dataset di test:\n "
    text=text+"-mae: "+str(mae(testY, result))+"\n-mape: "+str( mape(testY, result))+"\n"
    text=text+"numero epochs: "+str(epochs)+"\ntime lag: "+str(time_lag)+"\nbatch size: "+str(batch_size)+"\nvalidation split: "+str(validation_split)

    if(thisdropout!=0):
        text=text+"\nDropout rate: "+str(thisdropout)
    text=text+"\nSummary:\n"
    f.write(text)
    f.close()
    with open(nomeFileData,'a') as fh:
        my_model.summary(print_fn=lambda x: fh.write(x + '\n'))

def creaGrafico(history,option,epochs):
    x_axis =[]
    j=0
    if(epochs>=200):
        j=100
    while j<epochs:
        x_axis.append(j)
        j=j+1




    if(epochs>=200):
        plt.plot(x_axis,history.history[option][100:])
        plt.plot(x_axis,history.history['val_'+option][100:])
    else:
        plt.plot(x_axis,history.history[option][:])
        plt.plot(x_axis,history.history['val_'+option][:])

    plt.title('model '+option)
    plt.ylabel(option)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    return plt





if __name__=="__main__":
    main()