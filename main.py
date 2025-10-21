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

