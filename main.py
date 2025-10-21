import numpy as np
import cv2
import os
import shutil
import base64
from PIL import Image
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input  # type: ignore
from tensorflow.keras.models import Model # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import google.generativeai as genai

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Image Similarity API",
    description="Upload two images to get a similarity score, a visual heatmap (Grad-CAM), and an AI-generated explanation."
)

load_dotenv()

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
gradcam_model = base_model

gap_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_vector_output = gap_layer(base_model.output)
similarity_model = Model(inputs=base_model.input, outputs=feature_vector_output)

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

def score_function(output):
    return tf.reduce_mean(output, axis=(1, 2, 3))

def generate_heatmap_base64(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_resized_img = cv2.resize(img, (224, 224))
    
    img_array = np.expand_dims(original_resized_img, axis=0)
    preprocessed_array = preprocess_input(img_array.copy())
    
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=False)
    
    cam = gradcam(score=score_function, seed_input=preprocessed_array, penultimate_layer=-1)
    
    heatmap = np.uint8(cam[0] * 255)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(original_resized_img, 0.6, heatmap_color, 0.4, 0)
    
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', superimposed_img_rgb)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    return base64_image

def get_llm_explanation(img_path1, img_path2, score):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        
        prompt = f"""
        As an AI image analyst, I've determined two images have a similarity score of {score*100:.1f}%.
        Briefly explain why they received this score.
        1.  **Similarities:** Note the shared subject, composition, and background.
        2.  **Differences:** Pinpoint subtle details that prevent a 100% score.
        Keep the tone user-friendly and concise.
        """
        response = model.generate_content([prompt, img1, img2])
        return response.text
    except Exception as e:
        print(f"LLM explanation failed: {e}")
        return "The AI explanation could not be generated due to a server-side issue."

@app.post("/compare/")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    path1 = os.path.join(temp_dir, "image1.upload")
    path2 = os.path.join(temp_dir, "image2.upload")

    try:
        with open(path1, "wb") as buffer:
            shutil.copyfileobj(file1.file, buffer)
        with open(path2, "wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)
            
        score = get_similarity_score(path1, path2)
        
        heatmap1_b64 = generate_heatmap_base64(path1, gradcam_model)
        heatmap2_b64 = generate_heatmap_base64(path2, gradcam_model)
        
        explanation = get_llm_explanation(path1, path2, score)
        
        return JSONResponse(content={
            "similarity_percentage": round(float(score) * 100, 2),
            "ai_explanation": explanation,
            "heatmap_image1": f"data:image/png;base64,{heatmap1_b64}",
            "heatmap_image2": f"data:image/png;base64,{heatmap2_b64}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    finally:
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)

