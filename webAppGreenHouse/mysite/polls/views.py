
import os
from json import dumps

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.http import JsonResponse

from django.template import loader
from httpx import Response

from .applications import gestoreRichieste


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def home(request):

    return  render(request,"home.html");

def plot(request):

    return  render(request,"plot.html");

def autori(request):

    return  render(request,"autori.html")
def forecast(request):
    return  render(request,"forecast.html")

def forecastIstant(request):
    return render(request, "forecastIstant.html")

def forecastDays(request):
    return render(request,"forecastDays.html")

def test(request):
    # create data dictionary
    data="21°"

    data = dumps(data)
    return render(request, "test.html", {"data": data})


def testReq(request):
    var = request.GET.get('day')
    print(var)
    var=str(var)+" leo"
    return JsonResponse(os.getcwd(),safe=False)

def reqTemp(request):
    day= request.GET.get('day')
    time=request.GET.get('time')
    modelType=request.GET.get('model')
    righePred=gestoreRichieste.reqTemp(day, time,modelType)

    return JsonResponse(righePred,safe=False)

def reqTempDay(request):
    sensibility= request.GET.get('salto')
    day= request.GET.get('day')
    time=request.GET.get('time')
    modelType=request.GET.get('model')
    righePred=gestoreRichieste.getTemperaturaDay(sensibility,day,time,modelType)

    return JsonResponse(righePred,safe=False)

def reqTempDayMedia(request):
    precisioneSalto= request.GET.get('precisioneSalto')
    ogniTotMedia= request.GET.get('ogniTotMedia')
    day= request.GET.get('day')
    time=request.GET.get('time')
    modelType=request.GET.get('model')
    dimIntervallo = request.GET.get('dimIntervallo')
    righePred=gestoreRichieste.getTemperaturaMediaOgniTotIntervallo(precisioneSalto,ogniTotMedia,day,time,modelType,dimIntervallo)

    return JsonResponse(righePred,safe=False)

def reqTempMonthDay(request):
    sensibility= request.GET.get('salto')
    dataInizio= request.GET.get('dataInizio')
    dataFine=request.GET.get('dataFine')
    modelType=request.GET.get('model')
    righePred=gestoreRichieste.getTemperaturaMonthDay(sensibility,dataInizio,dataFine,modelType)

    return JsonResponse(righePred,safe=False)

def reqTempMonthDayMedia(request):
    precisioneSalto= request.GET.get('precisioneSalto')
    ogniTotMedia= request.GET.get('ogniTotMedia')
    dataInizio= request.GET.get('dataInizio')
    dataFine=request.GET.get('dataFine')
    modelType=request.GET.get('model')
    dimIntervallo = request.GET.get('dimIntervallo')
    righePred=gestoreRichieste.getTemperaturaMonthMedia(precisioneSalto,ogniTotMedia,dataInizio,dataFine,modelType,dimIntervallo)

    return JsonResponse(righePred,safe=False)






