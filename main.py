import numpy as np
import cv2
import os
import shutil
import base64
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_weights import ModelModifier # type: ignore

import google.generativeai as genai

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
gradcam_model = base_model
similarity_model = base_model  

gemini_api_key = os.getenv("gemini_api_key")
genai.configure(api_key=gemini_api_key)

def preprocess_for_similarity(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    return preprocess_input(img_array)

def extract_features(img_array):
    features = similarity_model.predict(img_array)
    return features.flatten()

def get_similarity_score(img_path1, img_path2):
    img1_processed = preprocess_for_similarity(img_path1)
    img2_processed = preprocess_for_similarity(img_path2)
    features1 = extract_features(img1_processed).reshape(1, -1)
    features2 = extract_features(img2_processed).reshape(1, -1)
    return cosine_similarity(features1, features2)[0][0]