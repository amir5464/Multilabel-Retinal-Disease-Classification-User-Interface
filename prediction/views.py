

from django.shortcuts import render
from django.http import JsonResponse
from tensorflow.keras.preprocessing.image import  img_to_array
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras.models import load_model
import os
#from tensorflow_addons.metrics import F1Score
import tensorflow_ranking as tfr
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from PIL import Image
import warnings
import joblib
# Suppress specific warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow_addons')
    import tensorflow_addons as tfa





warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

class HammingLoss(Metric):
    def __init__(self, threshold=0.5, name="hamming_loss", **kwargs):
        super(HammingLoss, self).__init__(name=name, **kwargs)
        self.threshold = tf.Variable(threshold, trainable=False, dtype=tf.float32)
        self.hamming_loss = self.add_weight(name="hl", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to binary labels based on threshold
        y_pred_binary = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        # Compute absolute differences
        tmp = tf.math.abs(y_true - y_pred_binary)

        # Compute mean over classes for each sample
        hl = tf.math.reduce_mean(tmp, axis=-1)

        # Update the Hamming loss and sample count
        self.hamming_loss.assign_add(tf.math.reduce_sum(hl))
        self.count.assign_add(tf.cast(tf.size(y_true) / tf.shape(y_true)[-1], tf.float32))

    def result(self):
        return self.hamming_loss / self.count

    def reset_state(self):
        # Reset the Hamming loss and sample count at the end of each epoch
        self.hamming_loss.assign(0.)
        self.count.assign(0.)



labels = ['DR', 'MH', 'ODC', 'TSLN', 'DN', 'ARMD', 'MYA', 'BRVO', 'ODP', 'ODE', 'LS', 'RS', 'CSR', 'CRS', 'CRVO', 'RPEC', 'MS', 'AION', 'ERM', 'AH', 'RT', 'EDN', 'PT', 'MHL', 'ST', 'TV', 'RP', 'other']

model_path1 = os.path.join('prediction', 'static', 'models','Classification', 'EfficientNetB4-Rfid-0.93.h5')
efficientnet_model = load_model(model_path1, custom_objects={'HammingLoss': HammingLoss()})
model_path3 = os.path.join('prediction', 'static', 'models','Classification', 'EfficientNetvs2-Rfid-0.95.h5')
efficientnet_modelvs2 = load_model(model_path3, custom_objects={'HammingLoss': HammingLoss()})
model_path5 = os.path.join('prediction', 'static', 'models','Classification', 'EfficientNetB3-Rfid-0.96.h5')
efficientnet_modelB3 = load_model(model_path5, custom_objects={'HammingLoss': HammingLoss()})
model_path11 = os.path.join('prediction', 'static', 'models','Ensemble','Classification', 'Stacking_ensemble2.h5')
ensemble_Classfication2 = load_model(model_path11, custom_objects={'HammingLoss': HammingLoss()})
model_path2 = os.path.join('prediction', 'static', 'models','Classification', 'EfficientNetB4-Rfid-0.92weight.h5')
efficientnet_model_weight = load_model(model_path2, custom_objects={'HammingLoss': HammingLoss()})

model_path4 = os.path.join('prediction', 'static', 'models','Classification', 'EfficientNetvs2-Rfid-0.93weight.h5')
efficientnet_modelvs2_weight = load_model(model_path4, custom_objects={'HammingLoss': HammingLoss()})

model_path6 = os.path.join('prediction', 'static', 'models','Classification', 'EfficientNetB3-Rfid-0.92weight.h5')
efficientnet_modelB3_weight = load_model(model_path6, custom_objects={'HammingLoss': HammingLoss()})
model_path10 = os.path.join('prediction', 'static', 'models','Ensemble','Classification', 'Stacking_ensemble1.h5')
ensemble_Classfication1 = load_model(model_path10, custom_objects={'HammingLoss': HammingLoss()})
model_path7 = os.path.join('prediction', 'static', 'models','Detector', 'Best_DenseNet201_detector.h5')
dectector_desnet = load_model(model_path7)
model_path8 = os.path.join('prediction', 'static', 'models','Detector', 'dectectorEfficientNetV2s.h5')
dectector_efficientnet = load_model(model_path8)
model_path9 = os.path.join('prediction', 'static', 'models','Ensemble','Detector', 'Stacking_ensembleDect.h5')
ensemble_dectector = load_model(model_path9)

loaded_models = []
for i in range(28):
    filename = f'prediction/static/models/Ensemble/Classification/Logistic Regression Models/logistic_regression_model_{i}.joblib'
    lr = joblib.load(filename)
    loaded_models.append(lr)


def index(request):
    data={}
    data ["section_heading"] = "Prediction Dashboard"
    return render(request, "prediction_form.html", data)



def dectector(image_paths):
    target_size=(224, 224)
    img = Image.open(image_paths)
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 

    dectector_desenet_pred = dectector_desnet.predict(img_array)
    dectector_efficientnet_pred = dectector_efficientnet.predict(img_array)
    dectector_stacking_image=np.hstack([dectector_desenet_pred, dectector_efficientnet_pred ])
    dectector_stacking_pred =ensemble_dectector(dectector_stacking_image)
    return dectector_stacking_pred


def preprocess_image(image, target_size):
    # Open the image using PIL
    img = Image.open(image)
    # Resize the image
    img = img.resize(target_size)
    # Convert to a NumPy array
    img_array = img_to_array(img)
    # Expand dimensions for batch processing (since model.predict expects a batch)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@method_decorator(csrf_exempt, name='dispatch')
def process_and_predict_image(request):
    try:
        uploaded_files = request.FILES.getlist('image')  # Get the uploaded image files as a list

        if not uploaded_files:
            response_data = {'success': False, 'message': 'No images uploaded'}
        else:
            # Process and predict each image individually
            prediction_results = []
            
            for uploaded_file in uploaded_files:
                dectector_stacking_pred=dectector(uploaded_file)
                if dectector_stacking_pred[0] >= 0.5:
                    predicted_labels_with_scores = []
                    y_pred_prob = ensemble_predict_stacking(uploaded_file)
                    # Create a list of label names and their corresponding prediction scores
                    label_scores = [(label, prob) for label, prob in zip(labels, y_pred_prob[0])]

                    # Filter labels based on a threshold (0.5 in this case) and store in the list
                    for label, prob in label_scores:
                        if prob >= 0.5:
                            predicted_labels_with_scores.append(f"{label}: {prob * 100:.2f}%")

                    # Display the labels and their prediction scores
                    prediction_results.append(predicted_labels_with_scores)
                    #y_pred = (y_pred_prob > 0.5).astype(int)
                    #predicted_labels = [label for idx, label in enumerate(labels) if y_pred[0][idx]]
                    #prediction_results.append(predicted_labels)

                else:
                    predicted_label = ['Normal'] 
                    prediction_results.append(predicted_label)
            response_data = {
                'success': True,
                'message': 'Predictions completed successfully',
                'prediction_results': prediction_results,
            }

    except Exception as e:
        response_data = {'success': False, 'message': str(e)}

    return JsonResponse(response_data)

def EyePrediction(request):
    data = {}
    data["section_heading"] = "Prediction Dashboard"

    if request.method == 'POST':
        try:
            uploaded_files = request.FILES.getlist('image')  

            if not uploaded_files:
                response_data = {'success': False, 'message': 'No images uploaded'}
            else:

                return JsonResponse({'success': True, 'message': 'Images uploaded successfully'})

        except Exception as e:
            response_data = {'success': False, 'message': str(e)}

    return render(request, "prediction_imageclassify.html", data)

def predict_stacking(image_paths):

    classification_efficientnet_image = preprocess_image(image_paths, target_size=(380, 380))
    classification_efficientnet_pred = efficientnet_model.predict(classification_efficientnet_image )
    classification_efficientnetvs2_image = preprocess_image(image_paths, target_size=(224, 224))
    classification_efficientnetvs2_pred = efficientnet_modelvs2 .predict(classification_efficientnetvs2_image ) 
    classification_efficientnetB3_image = preprocess_image(image_paths, target_size=(300, 300))
    classification_efficientnetB3_pred = efficientnet_modelB3 .predict(classification_efficientnetB3_image ) 
    ensemble_predictions2 = np.hstack((classification_efficientnet_pred, classification_efficientnetvs2_pred,classification_efficientnetB3_pred))
    stacking2_predictions = ensemble_Classfication2.predict(ensemble_predictions2)
    return stacking2_predictions 

def predict_stacking_weight(image_paths):

    classification_efficientnetweight_image = preprocess_image(image_paths, target_size=(380, 380))
    classification_efficientnetweight_pred = efficientnet_model_weight.predict(classification_efficientnetweight_image )

    classification_efficientnetvs2weight_image = preprocess_image(image_paths, target_size=(224, 224))
    classification_efficientnetvs2weight_pred = efficientnet_modelvs2_weight.predict(classification_efficientnetvs2weight_image )

    classification_efficientnetB3weight_image = preprocess_image(image_paths, target_size=(300, 300))
    classification_efficientnetB3weight_pred = efficientnet_modelB3_weight .predict(classification_efficientnetB3weight_image ) 
    ensemble_predictions1 = np.hstack((classification_efficientnetweight_pred, classification_efficientnetvs2weight_pred,classification_efficientnetB3weight_pred))
    
    stacking1_predictions = ensemble_Classfication1.predict(ensemble_predictions1)
 
    return stacking1_predictions 

def ensemble_predict_stacking(image_paths):

    stacking1_predictions=predict_stacking_weight(image_paths)
    stacking2_predictions=predict_stacking(image_paths)
    ensemble_stacking = np.hstack((stacking1_predictions, stacking2_predictions))
    final_predictions = []
    for lr in loaded_models:
        pred = lr.predict_proba(ensemble_stacking)[:, 1]
        final_predictions.append(pred)
    return np.array(final_predictions).T