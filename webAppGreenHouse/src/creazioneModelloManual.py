import os
from matplotlib import pyplot as plt
from sklearn.metrics import (mean_absolute_error as mae)
from sklearn.metrics import (mean_absolute_percentage_error as mape)
from gestioneDati import *
from standardModelli import model
from standardModelli import modelCNN_LSTM
from standardModelli import modelLSTM_CNN
from standardModelli import modelCLPARALLEL


#FILE_MODELLO_KERAS = './dati/keras_st4_model.h5'
FILE_DATI_CSV = './dati/Dataset_sens_'
FILE_TEST_CSV = './dati/Test_sens_4.csv'
FILE_MODELLO_KERAS ='./dati/my_model.h5'
dirModelli='./modelli'
time_lag = 7
epochs = 300

thisdropout = 0.005
batch_size =70
validation_split = 0.2

def main():
    tipo=None
    sensore=None
    shuffle=None
    #Richieste utente
    while(tipo==None):
        tipo=input("inserisci tipo di modello (1-LSTM, 2-CNN_LSTM, 3-LSTM_CNN, 4-CNN_LSTM_Par): ")
        if(tipo=="1"):
            tipo="LSTM"
        elif(tipo=="2"):
            tipo="CNN_LSTM"
        elif(tipo=="3"):
            tipo="LSTM_CNN"
        elif(tipo=="4"):
            tipo="CNN_LSTM_PAR"
        else:
            print("valore non valido, riprova")
            tipo=None
    while(sensore==None):
        sensore=input("inserisci sensore (da 1 a 8): ")

        if(sensore.isnumeric()):
            val=int(sensore)
            if(val<1 or val>8):
                print("valore non valido, riprova")
                sensore=None
        else:
            print("valore non valido, riprova")
            sensore=None
    while(shuffle==None):
        shuffle=input("vuoi che valori vengano mescolati(shuffle)(y/n)?")
        if(shuffle=="y"):
            shuffle=True
        elif(shuffle=="n"):
            shuffle=False
        else:
            print("valore errato, riprova.")
            shuffle=None


    #Modello+dati scelti
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

    #Creazione cartella modello
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

    #Salvataggio e training modello
    history =my_model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,
                                      validation_split=validation_split, verbose=1)
    my_model.save(nomeFileModello)


    #Salvataggio dati modello (grafici+txt)
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
    #Funzione di supporto per la creazione del grafico considerato l'history+numero epochs
    #option = mae / mape (stringhe)
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