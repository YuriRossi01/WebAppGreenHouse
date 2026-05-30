import random


import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from constants import data_header, data_headers


def suddivisioneDatasetXY(dataset, dataHeader):
    # dato dataset in input, ritorna dataX,dataY ricavati dal
    # dataset dato.
    #Del dataset si prendono i valori corrispondenti all'header preso dalle costanti(file)
    #Si considera valori Y quelli all'ultima posizione del dataset (su riga)
    X = []
    Y = []
    seq_length = len(dataHeader)
    for i in range(len(dataset)):
        X.append(dataset[i][:seq_length - 1])
        Y.append(dataset[i][seq_length - 1:])
    X = numpy.array(X)
    Y = numpy.array(Y)
    return X, Y


def trasformationTimeStep(arrayData, timeStep):
    #dato array (numpy) con shape(nCampioni,nValori),
    # ritorna array con shape(nCampioni,time_lag,nValori)
    #time_lag costante
    #"si rende array divisibile per time_lag"
    #es: (0,1,2,3,4);(1,2,3,4,5)...
    arrayDataTimeStep= []
    for i in range(0,len(arrayData)-(timeStep-1)):
        group=[]
        for j in range(0,timeStep):
            group.append(arrayData[i+j])
        arrayDataTimeStep.append(group)
    arrayDataTimeStep=numpy.asarray(arrayDataTimeStep)


    return arrayDataTimeStep


def doShuffle(x, y):
    #Dati x,y randomizza posizioni e ritorna array rimescolati
    zipped = list(zip(x, y))
    random.Random(40938233124).shuffle(zipped)
    x, y = zip(*zipped)
    x = numpy.array(x)
    y = numpy.array(y)
    return x, y
def getTrainXYTestXY(nomeFile, dataDescr, timeStep, shuffle=False ):
    #dato nome file e dataDescr (stringa che indica "tipo" di data header da prendere
    #da costanti e timeStep, ritorna trainX,trainY,testX,testY
    #Dati già normalizzati

    #dataset
    datasetTemp = pandas.read_csv(nomeFile)
    dataHeader = data_headers.get(dataDescr)
    indexTimesstamp = dataHeader.index('timestamp_normalizzato')
    datasetOrig = datasetTemp[dataHeader].values
    for i in range(len(datasetOrig)):
        val =(datasetOrig[i][indexTimesstamp][len(datasetOrig[i][indexTimesstamp]) - 1])
        datasetOrig[i][indexTimesstamp] =  int(val)*15
    datasetOrig = numpy.asarray(datasetOrig).astype(numpy.float32)


    dataset=datasetOrig.copy()
    trainSize = int(0.8 * len(dataset))

    #suddivisione train,test
    train = dataset[0:trainSize]
    test = dataset[trainSize:]
    (trainX, trainY) = suddivisioneDatasetXY(train, dataHeader)
    (testX, testY) = suddivisioneDatasetXY(test, dataHeader)

    if shuffle:
        doShuffle(trainX, trainY)
        doShuffle(testX, testY)

    #trasformazione con timeStep
    trainX = trasformationTimeStep(trainX, timeStep)
    testX = trasformationTimeStep(testX, timeStep)


    return trainX, trainY, testX, testY






