import codecs
import datetime
import json
import os
import random
import tensorflow
from logging import exception
from pathlib import Path
from turtle import clone

import keras
import numpy
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dirModelli='static/modelliKeras'
batch_size = 1
time_lag = 7




def getTemperaturaMonthMedia(precisioneSalto,puntiMedia,dataInizio,dataFine,modelType,intervallo,local=False):
    dataInizio= datetime.datetime(int(dataInizio.split("-")[0]), int(dataInizio.split("-")[1]), int(dataInizio.split("-")[2]),0,0,0,0)
    dataFine=datetime.datetime(int(dataFine.split("-")[0]), int(dataFine.split("-")[1]), int(dataFine.split("-")[2]),0,0,0,0)

    listaDate=[]
    valPrec=int(precisioneSalto.split(".")[0])
    unitaMisuraPrec=(precisioneSalto.split(".")[1])
    if(unitaMisuraPrec=="h"):
        valPrec=valPrec*60
    timedeltaPrec=datetime.timedelta(minutes=valPrec)

    valPunt=int(puntiMedia.split(".")[0])
    unitaMisuraPunt=(puntiMedia.split(".")[1])
    if(unitaMisuraPunt=="h"):
        valPunt=valPunt*60

    nValoriFinali= ((24*60*(dataFine-dataInizio).days)//valPunt)+1




    intervallo=int(intervallo)
    dataTemp=dataInizio

    while(dataTemp<=dataFine):
        listaDate.append(dataTemp)
        for i in range(1,intervallo+1):
            listaDate.append(dataTemp+(i*timedeltaPrec))
            listaDate.append(dataTemp-(i*timedeltaPrec))
        dataTemp=dataTemp+datetime.timedelta(minutes=valPunt)


    gruppoRigheTimeStep=[]

    for data in listaDate:
        rigaTimeStep=[]
        for i in range(0,time_lag):
            dataT=(data+i*timedeltaPrec)

            rigatemp=(getRigaDef(dataT))
            rigaTimeStep.append(rigatemp)

        gruppoRigheTimeStep.append(rigaTimeStep)
    arrayRighe = numpy.asarray(gruppoRigheTimeStep)



    filenameBase = dirModelli + '/' + "LSTM" + "_"+"model"+"_"
    filename=""


    #filename=getNomeFileModello(modelType,sensore)

    allResults=[]
    for sensore in range(1,9):
        filename=getNomeFileModello(modelType,sensore)
        my_file = Path(filename)
        if not(my_file.is_file()):
            raise ValueError("nome file non valido")

        my_model = keras.saving.load_model(my_file)
        result = my_model.predict(arrayRighe, batch_size=batch_size)

        medie=[]
        posAbs=0
        for i in range(0,nValoriFinali):
            media=0
            countMedie=0
            for posRel in range(0,intervallo*2+1):
                media=media+result[posAbs+posRel]
                countMedie=countMedie+1
            media=media/countMedie
            medie.append(media)
            posAbs=posAbs+intervallo*2+1

        allResults.append(medie)
    allResults=np.asarray(allResults)
    jsonData = json.dumps(allResults.tolist())
    return jsonData

def getTemperaturaMonthDay(misuraSalto,dataInizio,dataFine,modelType,local=False):
    dataInizio= datetime.datetime(int(dataInizio.split("-")[0]), int(dataInizio.split("-")[1]), int(dataInizio.split("-")[2]),0,0,0,0)
    dataFine=datetime.datetime(int(dataFine.split("-")[0]), int(dataFine.split("-")[1]), int(dataFine.split("-")[2]),0,0,0,0)
    listaDate=[]

    val=int(misuraSalto.split(".")[0])
    unitaMisura=(misuraSalto.split(".")[1])

    if(unitaMisura=="m"):
        timedelta= datetime.timedelta(minutes=val)
    if(unitaMisura=="h"):
        timedelta=datetime.timedelta(hours=val)

    i=0
    while((dataInizio+(i*timedelta))<=dataFine):
        listaDate.append(dataInizio+(i*timedelta))
        i=i+1

    gruppoRigheTimeStep=[]
    for data in listaDate:
        rigaTimeStep=[]
        for i in range(0,time_lag):
            dataT=(data+i*timedelta)
            rigatemp=(getRigaDef(dataT))
            rigaTimeStep.append(rigatemp)
        gruppoRigheTimeStep.append(rigaTimeStep)
    arrayRighe = numpy.asarray(gruppoRigheTimeStep)



    #creazione nome file

    allResults=[]
    for sensore in range(1,9):
        filename=getNomeFileModello(modelType,sensore)
        my_file = Path(filename)
        if not(my_file.is_file()):
            raise ValueError("nome file non valido")

        my_model = keras.saving.load_model(my_file)
        result = my_model.predict(arrayRighe, batch_size=batch_size)
        allResults.append(result)
    allResults=np.asarray(allResults)

    jsonData = json.dumps(allResults.tolist())
    return jsonData

def getTemperaturaMediaOgniTotIntervallo(precisioneSalto,puntiMedia,data,tempo,modelType,intervallo,local=False):
    dataInizio= datetime.datetime(int(data.split("-")[0]), int(data.split("-")[1]), int(data.split("-")[2]),int(tempo.split(":")[0]),int(tempo.split(":")[1]),0,0)
    dataFine=dataInizio+datetime.timedelta(days=1)

    listaDate=[]
    valPrec=int(precisioneSalto.split(".")[0])
    unitaMisuraPrec=(precisioneSalto.split(".")[1])
    if(unitaMisuraPrec=="h"):
        valPrec=valPrec*60
    timedeltaPrec=datetime.timedelta(minutes=valPrec)

    valPunt=int(puntiMedia.split(".")[0])
    unitaMisuraPunt=(puntiMedia.split(".")[1])
    if(unitaMisuraPunt=="h"):
        valPunt=valPunt*60

    nValoriFinali= ((24*60)//valPunt)+1




    intervallo=int(intervallo)

    dataTemp=dataInizio

    while(dataTemp<=dataFine):
        listaDate.append(dataTemp)
        for i in range(1,intervallo+1):
            listaDate.append(dataTemp+(i*timedeltaPrec))
            listaDate.append(dataTemp-(i*timedeltaPrec))
        dataTemp=dataTemp+datetime.timedelta(minutes=valPunt)


    gruppoRigheTimeStep=[]

    for data in listaDate:
        rigaTimeStep=[]
        for i in range(0,time_lag):
            dataT=(data+i*timedeltaPrec)

            rigatemp=(getRigaDef(dataT))
            rigaTimeStep.append(rigatemp)

        gruppoRigheTimeStep.append(rigaTimeStep)
    arrayRighe = numpy.asarray(gruppoRigheTimeStep)


    allResults=[]
    for sensore in range(1,9):
        filename=getNomeFileModello(modelType,sensore)
        my_file = Path(filename)
        if not(my_file.is_file()):
            raise ValueError("nome file non valido")

        my_model = keras.saving.load_model(my_file)
        result = my_model.predict(arrayRighe, batch_size=batch_size)

        medie=[]
        posAbs=0
        for i in range(0,nValoriFinali):
            media=0
            countMedie=0
            for posRel in range(0,intervallo*2+1):
                media=media+result[posAbs+posRel]
                countMedie=countMedie+1
            media=media/countMedie
            medie.append(media)
            posAbs=posAbs+intervallo*2+1

        #medie = np.asarray(medie).tolist()
        allResults.append(medie)
    allResults=np.asarray(allResults)
    jsonData = json.dumps(allResults.tolist())
    return jsonData

def getTemperaturaDay(misuraSalto,data,tempo,modelType,local=False):
    dataInizio= datetime.datetime(int(data.split("-")[0]), int(data.split("-")[1]), int(data.split("-")[2]),int(tempo.split(":")[0]),int(tempo.split(":")[1]),0,0)
    dataFine=dataInizio+datetime.timedelta(days=1)
    listaDate=[]

    val=int(misuraSalto.split(".")[0])
    unitaMisura=(misuraSalto.split(".")[1])

    if(unitaMisura=="m"):
        timedelta= datetime.timedelta(minutes=val)
    if(unitaMisura=="h"):
        timedelta=datetime.timedelta(hours=val)

    i=0
    while((dataInizio+(i*timedelta))<=dataFine):
        listaDate.append(dataInizio+(i*timedelta))
        i=i+1

    gruppoRigheTimeStep=[]
    for data in listaDate:
        rigaTimeStep=[]
        for i in range(0,time_lag):
            dataT=(data+i*timedelta)
            rigatemp=(getRigaDef(dataT))
            rigaTimeStep.append(rigatemp)
        gruppoRigheTimeStep.append(rigaTimeStep)
    arrayRighe = numpy.asarray(gruppoRigheTimeStep)






    allResults=[]
    for sensore in range(1,9):
        filename=getNomeFileModello(modelType,sensore)
        my_file = Path(filename)
        if not(my_file.is_file()):
            raise ValueError("nome file non valido")
        my_model = keras.saving.load_model(my_file)
        result = my_model.predict(arrayRighe, batch_size=batch_size)
        allResults.append(result)
    allResults = numpy.asarray(allResults)

    jsonData = json.dumps(allResults.tolist())
    return jsonData

def  reqTemp(data,tempo,modelType,extra=False):

    if(type(data)==str):
        dataInizio= datetime.datetime(int(data.split("-")[0]), int(data.split("-")[1]), int(data.split("-")[2]),int(tempo.split(":")[0]),int(tempo.split(":")[1]),0,0)
    else:
        dataInizio=data
    dates=[]
    for i in range(0,time_lag):
        dates.append(dataInizio+datetime.timedelta(minutes=15*i))
    gruppoRigheTimeStep=[]
    j = 0
    for date in dates:
        j = j+1
        print(j)
        rigatemp=(getRigaDef(date))
        gruppoRigheTimeStep.append(rigatemp)
    arrayRighe = []
    arrayRighe.append(gruppoRigheTimeStep)


    arrayRighe = numpy.asarray(arrayRighe)



    #creazione nome file


    # filenameBase = dirModelli + '/' + "LSTM" + "_"+"model"+"_"
    filename=""
    allResults=[]
    for sensore in range(1,9):
        filename=getNomeFileModello(modelType,sensore)



        my_file = Path(filename)
        if not(my_file.is_file()):
            raise ValueError("nome file non valido")

        my_model = keras.saving.load_model(my_file)
        result = my_model.predict(arrayRighe, batch_size=batch_size)





        allResults.append(result)
    allResults = numpy.asarray(allResults)
    jsonData = json.dumps(allResults.tolist())

    return jsonData
def getDaysMonth(data):

    if data.month==11 or data.month==4 or data.month==6 or data.month==9:
        return 30
    elif data.month==2:
        if data.year%400==0 or (data.year%4==0 and not(data.year%100==0)):
            return 29
        else:
            return 28
    else:
        return 31
def getNomeFileModello(modelType,sensore):
    filename = dirModelli + '/'

    if(modelType=="CNNLSTM"):
        filename=filename+'modelliCNNLSTM'+'/'+'CNN_LSTM_model_'
    elif (modelType=="LSTMCNN"):
        filename=filename+'modelliLSTMCNN'+'/'+'LSTM_CNN_model_'
    elif (modelType=="LSTM"):
        filename=filename+'modelliLSTM'+'/'+'LSTM_model_'
    elif (modelType=="CNNLSTMPar"):
        filename=filename+'modelliCNNLSTMPar'+'/'+'CNN_LSTM_PAR_model_'
    else:
        raise ValueError("tipo modello mancante o non valido")
    filename=filename+str(sensore)+'/model.keras'
    return filename

def getRighePred(dataInizio,dataFine,passoTemporale):
    righe=[]


    rigaDef = getRigaDef(dataInizio)
    righe.append(rigaDef)

    if(passoTemporale=="timeStep"):
        passo=datetime.timedelta(minutes=15)
    elif(passoTemporale=="hour"):
        passo=datetime.timedelta(hours=1)
    elif(passoTemporale=="day"):
        passo=datetime.timedelta(days=1)

    data=dataInizio

    while(not(data.date()==dataFine.date() and data.time()==dataFine.time())):
        rigaCopia=rigaDef.copy()
        data+=passo

        #'anno_acquisizione',
        rigaCopia[1] = data.year
        #'mese_acquisizione',
        rigaCopia[2] = (data.month)
        #'settimana_acquisizione_su_anno',
        rigaCopia[3]=int(data.strftime("%W"))
        #'giorno_acquisizione_su_anno',
        rigaCopia[4]=int(data.strftime("%j"))
        #'giorno_acquisizione_su_mese',
        rigaCopia[5]=int(data.strftime("%d"))
        #'giorno_acquisizione_su_settimana',
        rigaCopia[6]=int(data.strftime("%w"))
        #'ora_acquisizione',
        rigaCopia[7]=data.hour
        #timestamp
        rigaCopia[8]=data.minute
        righe.append(rigaCopia)
    for i in range(0,7):
        rigaCopia=rigaDef.copy()
        data+=passo
        #'anno_acquisizione',
        rigaCopia[1] = data.year
        #'mese_acquisizione',
        rigaCopia[2] = (data.month)
        #'settimana_acquisizione_su_anno',
        rigaCopia[3]=int(data.strftime("%W"))
        #'giorno_acquisizione_su_anno',
        rigaCopia[4]=int(data.strftime("%j"))
        #'giorno_acquisizione_su_mese',
        rigaCopia[5]=int(data.strftime("%d"))
        #'giorno_acquisizione_su_settimana',
        rigaCopia[6]=int(data.strftime("%w"))
        #'ora_acquisizione',
        rigaCopia[7]=data.hour
        #timestamp
        rigaCopia[8]=data.minute
        righe.append(rigaCopia)


    return righe
def getRigaDef(data):
    #sens 1
    rigaDef=[]

    #'distanza_da_centralina_cm',
    rigaDef.append(1679)
    #'anno_acquisizione',
    rigaDef.append(data.year)

    #'mese_acquisizione',
    rigaDef.append(data.month)
    #'settimana_acquisizione_su_anno',
    rigaDef.append(int(data.strftime("%W")))

    #'giorno_acquisizione_su_anno',
    rigaDef.append(int(data.strftime("%j")))
    #'giorno_acquisizione_su_mese',
    rigaDef.append(int(data.strftime("%d")))

    #'giorno_acquisizione_su_settimana',
    rigaDef.append(int(data.strftime("%w")))
    #'ora_acquisizione',
    rigaDef.append(data.hour)

    #'timestamp_normalizzato',
    rigaDef.append(data.minute)
    #'evja_temp',
    rigaDef.append(None)

    #'Barometer_HPa',
    rigaDef.append(-1)
    #'Temp__C',
    rigaDef.append(-1)

    #'HighTemp__C',
    rigaDef.append(-1)
    #'LowTemp__C',
    rigaDef.append(-1)
    #'Hum__',
    rigaDef.append(-1)
    #'DewPoint__C',
    rigaDef.append(-1)
    #'WetBulb__C',
    rigaDef.append(-1)
    #'WindSpeed_Km_h',
    rigaDef.append(-1)
    #'WindDirection_',
    rigaDef.append(-1)
    #'WindRun_Km',
    rigaDef.append(-1)
    #'HighWindSpeed_Km_h',
    rigaDef.append(-1)
    #'HighWindDirection_',
    rigaDef.append(-1)
    #'WindChill__C',
    rigaDef.append(-1)
    #'HeatIndex__C',
    rigaDef.append(-1)
    #'THWIndex__C',
    rigaDef.append(-1)
    #'THSWIndex__C',
    rigaDef.append(-1)
    #'Rain_Mm',
    rigaDef.append(-1)
    #'RainRate_Mm_h',
    rigaDef.append(-1)
    #'SolarRad_W_m_2',
    rigaDef.append(-1)
    #'SolarEnergy_Ly',
    rigaDef.append(-1)
    #'HighSolarRad_W_m_2',
    rigaDef.append(-1)
    #'ET_Mm',
    rigaDef.append(-1)
    #'UVIndex_',
    rigaDef.append(-1)
    #'UVDose_MEDs_',
    rigaDef.append(-1)
    #'HighUVIndex_',
    rigaDef.append(-1)
    #'HeatingDegreeDays',
    rigaDef.append(-1)
    #'CoolingDegreeDays',
    rigaDef.append(-1)
    #'Humidity__RH_',
    rigaDef.append(-1)
    #'Solar_klux_',
    rigaDef.append(-1)


    return getDatiFasulli(rigaDef,data)

def getDatiFasulli(rigaDef,data):
    # for i in range(9,len(rigaDef)):


    #'evja_temp',
    rigaDef[9]=getTempFasulla(data)
    #'Barometer_HPa', circa media
    rigaDef[10]=1000
    #'Temp__C',
    rigaDef[11]=getTempFasulla(data)

    #'HighTemp__C',
    rigaDef[12]=rigaDef[9]+0.5
    #'LowTemp__C',
    rigaDef[13]=rigaDef[9]-0.5
    #'Hum__', circa media
    rigaDef[14]=70
    #'DewPoint__C',
    rigaDef[15]=getTempFasulla(data)
    if(rigaDef[15]>20):
        rigaDef[15]=20
    if(rigaDef[15]<-1):
        rigaDef[15]=-1
    #'WetBulb__C',
    rigaDef[16]=getTempFasulla(data)
    if(rigaDef[16]>20):
        rigaDef[16]=20
    if(rigaDef[16]<-1):
        rigaDef[16]=-1
    #'WindSpeed_Km_h',
    rigaDef[17]=random.uniform(0.0,10.0)
    #'WindDirection_',
    rigaDef[18]=random.randint(1,14)
    #'WindRun_Km',
    rigaDef[19]=1.5
    #'HighWindSpeed_Km_h',
    rigaDef[20]=15
    #'HighWindDirection_',
    rigaDef[21]=0
    #'WindChill__C',
    rigaDef[22]=rigaDef[11]
    #'HeatIndex__C',
    rigaDef[23]=rigaDef[11]
    #'THWIndex__C',
    rigaDef[24]=rigaDef[11]
    #'THSWIndex__C',
    rigaDef[25]=rigaDef[11]
    #'Rain_Mm',
    rigaDef[26]=0
    #'RainRate_Mm_h',
    rigaDef[27]=0
    #'SolarRad_W_m_2',
    rigaDef[28]=random.randint(100,600)
    #'SolarEnergy_Ly',
    rigaDef[29]=rigaDef[28]/50
    #'HighSolarRad_W_m_2',
    rigaDef[30]=rigaDef[28]
    #'ET_Mm',
    rigaDef[31]=0
    #'UVIndex_',
    rigaDef[32]=0
    #'UVDose_MEDs_',
    rigaDef[33]=0
    #'HighUVIndex_',
    rigaDef[34]=0
    #'HeatingDegreeDays',
    rigaDef[35]=0
    #'CoolingDegreeDays',
    rigaDef[36]=0
    #'Humidity__RH_',
    rigaDef[37]=70
    #'Solar_klux_',
    rigaDef[38]=random.randint(20,60)

    return rigaDef

def getTempFasulla(data):

    if(data.month>=1 and data.month<=2):
        if(data.hour>=10 and data.hour<=14):
            return random.uniform(3.0,5.0)
        if(data.hour>=15 and data.hour<=18):
            return random.uniform(2.0,4.0)
        if(data.hour>=19 and data.hour<=22):
            return random.uniform(0.0,2.0)
        if(data.hour>=23 or data.hour<=4):
            return random.uniform(-5.0,0.0)
        if(data.hour>=5 and data.hour<=9):
            return random.uniform(-3.0,1.0)

    if(data.month>=3 and data.month<=4):
        if(data.hour>=10 and data.hour<=14):
            return random.uniform(13.0,14.0)
        if(data.hour>=15 and data.hour<=18):
            return random.uniform(10.0,13.0)
        if(data.hour>=19 and data.hour<=22):
            return random.uniform(6.0,8.0)
        if(data.hour>=23 or data.hour<=4):
            return random.uniform(4.0,6.0)
        if(data.hour>=5 and data.hour<=9):
            return random.uniform(8.0,10.0)
    if(data.month>=5 and data.month<=6):
        if(data.hour>=10 and data.hour<=14):
            return random.uniform(22.0,25.0)
        if(data.hour>=15 and data.hour<=18):
            return random.uniform(20.0,24.0)
        if(data.hour>=19 and data.hour<=22):
            return random.uniform(19.0,23.0)
        if(data.hour>=23 or data.hour<=4):
            return random.uniform(14.0,16.0)
        if(data.hour>=5 and data.hour<=9):
            return random.uniform(15.0,20.0)
    if(data.month>=7 and data.month<=8):
        if(data.hour>=10 and data.hour<=14):
            return random.uniform(28.0,32.0)
        if(data.hour>=15 and data.hour<=18):
            return random.uniform(27.0,30.0)
        if(data.hour>=19 and data.hour<=22):
            return random.uniform(20.0,25.0)
        if(data.hour>=23 or data.hour<=4):
            return random.uniform(18.0,20.0)
        if(data.hour>=5 and data.hour<=9):
            return random.uniform(20.0,25.0)
    if(data.month>=9 and data.month<=10):
        if(data.hour>=10 and data.hour<=14):
            return random.uniform(18.0,22.0)
        if(data.hour>=15 and data.hour<=18):
            return random.uniform(16.0,20.0)
        if(data.hour>=19 and data.hour<=22):
            return random.uniform(17.0,19.0)
        if(data.hour>=23 or data.hour<=4):
            return random.uniform(15.0,17.0)
        if(data.hour>=5 and data.hour<=9):
            return random.uniform(17.0,19.0)
    if(data.month>=11 and data.month<=12):
        if(data.hour>=10 and data.hour<=14):
            return random.uniform(6.0,10.0)
        if(data.hour>=15 and data.hour<=18):
            return random.uniform(5.0,8.0)
        if(data.hour>=19 and data.hour<=22):
            return random.uniform(3.0,5.0)
        if(data.hour>=23 or data.hour<=4):
            return random.uniform(0.0,5.0)
        if(data.hour>=5 and data.hour<=9):
            return random.uniform(2.0,6.0)