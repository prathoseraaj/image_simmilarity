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
import google.generativeai as genai

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Image Similarity API",
    description="Upload two images to get a similarity score, a visual heatmap (Grad-CAM), and an AI-generated explanation."
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

def generate_similarity_heatmaps(img_path1, img_path2, model):

    img1 = cv2.imread(img_path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_resized = cv2.resize(img1, (224, 224))
    
    img2 = cv2.imread(img_path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2_resized = cv2.resize(img2, (224, 224))
    
    img1_array = np.expand_dims(img1_resized, axis=0)
    img2_array = np.expand_dims(img2_resized, axis=0)
    
    img1_prep = preprocess_input(img1_array.copy())
    img2_prep = preprocess_input(img2_array.copy())
    
    feature_model = Model(inputs=model.input, 
                         outputs=model.get_layer('conv5_block3_out').output)
    
    features1 = feature_model.predict(img1_prep)[0]
    features2 = feature_model.predict(img2_prep)[0]
    
    features1_norm = features1 / (np.linalg.norm(features1, axis=-1, keepdims=True) + 1e-8)
    features2_norm = features2 / (np.linalg.norm(features2, axis=-1, keepdims=True) + 1e-8)
    
    similarity_map = np.sum(features1_norm * features2_norm, axis=-1)
    similarity_map = np.maximum(similarity_map, 0)
    
    similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-8)
    
    similarity_map = cv2.resize(similarity_map, (224, 224))
    
    sim_heatmap = np.uint8(similarity_map * 255)
    sim_heatmap_color = cv2.applyColorMap(sim_heatmap, cv2.COLORMAP_JET)
    sim_overlay1 = cv2.addWeighted(img1_resized, 0.6, sim_heatmap_color, 0.4, 0)
    
    sim_overlay2 = cv2.addWeighted(img2_resized, 0.6, sim_heatmap_color, 0.4, 0)
    
    _, buffer1 = cv2.imencode('.png', cv2.cvtColor(sim_overlay1, cv2.COLOR_RGB2BGR))
    sim1_b64 = base64.b64encode(buffer1).decode('utf-8')
    
    _, buffer2 = cv2.imencode('.png', cv2.cvtColor(sim_overlay2, cv2.COLOR_RGB2BGR))
    sim2_b64 = base64.b64encode(buffer2).decode('utf-8')
    
    return {
        'similarity_img1': sim1_b64,
        'similarity_img2': sim2_b64
    }

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
        
        heatmaps = generate_similarity_heatmaps(path1, path2, base_model)
        
        explanation = get_llm_explanation(path1, path2, score)
        
        return JSONResponse(content={
            "similarity_score": float(score),
            "heatmap_image1": heatmaps['similarity_img1'],
            "heatmap_image2": heatmaps['similarity_img2'],
            "insights": explanation
        })


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    finally:
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)

