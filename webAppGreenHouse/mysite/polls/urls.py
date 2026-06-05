from django.urls import path

from . import views

urlpatterns = [

    #pagine
    path("home/",views.home),
    path("plot/",views.plot),
    path("reqTempMonthDayMedia/",views.reqTempMonthDayMedia),
    path("autori/",views.autori),
    path("forecast/",views.forecast),
    path("forecastIstant/", views.forecastIstant),
    path("forecastDays/", views.forecastDays),
    path("darkMode/", views.darkMode),

    #richieste
    path("test/",views.test),
    path("testReq/",views.testReq),
    path("reqTemp/",views.reqTemp),
    path("reqTempDay/",views.reqTempDay),
    path("reqTempDayMedia/",views.reqTempDayMedia),
    path("reqTempMonthDay/",views.reqTempMonthDay),


]