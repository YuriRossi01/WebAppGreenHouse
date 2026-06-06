from django.urls import path
from django.contrib.staticfiles.storage import staticfiles_storage
from django.views.generic.base import RedirectView
from django.conf import settings
from . import views

urlpatterns = [
    #favicon.ico
    path(
        'favicon.ico',
        RedirectView.as_view(url=staticfiles_storage.url('images/favicon.png')),
        name='favicon'
    ),


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