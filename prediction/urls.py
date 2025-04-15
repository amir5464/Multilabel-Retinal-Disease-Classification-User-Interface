 
from django.urls import path
from prediction import views


urlpatterns = [
   
   
    path('', views.index, name='prediction_form'),
    path('process_and_predict_image/', views.process_and_predict_image, name='process_and_predict_image'),
    path('EyePrediction/', views.EyePrediction, name='EyePrediction'),
 
]
