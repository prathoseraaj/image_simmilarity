import numpy as np
import cv2
import os
import shutil
import base64
import traceback
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import torch

# FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware 


from transformers import CLIPProcessor, CLIPModel 

# Grad-CAM
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image

class SimilarityTarget:
    def __init__(self, target_features):
        self.target = target_features

    def __call__(self, model_output):
        return torch.nn.functional.cosine_similarity(
            model_output, 
            self.target, 
            dim=-1
        )

app = FastAPI(
    title="Image Similarity API (CLIP + Grad-CAM Version)", # Title updated
    description="Upload two images to get a semantic similarity score (CLIP), visual heatmaps (Grad-CAM), and an AI-generated explanation."
)

origins = ["http://localhost:3000"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

gemini_api_key = os.getenv("gemini_api_key")
genai.configure(api_key=gemini_api_key)

class CLIPModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values)

def reshape_transform(tensor):
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    
    result = tensor[:, 1:, :].reshape(tensor.size(0), 7, 7, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

target_layer = [clip_model.vision_model.encoder.layers[-1].layer_norm1]

cam = GradCAM(
    model=CLIPModelWrapper(clip_model),
    target_layers=target_layer,
    reshape_transform=reshape_transform
)

def get_similarity_score(img_path1, img_path2):

    image1 = Image.open(img_path1)
    image2 = Image.open(img_path2)
    
    inputs = clip_processor(
        images=[image1, image2],
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features /= features.norm(p=2, dim=-1, keepdim=True)
        similarity = float((features[0] @ features[1].T).item())
        
    return similarity

def generate_similarity_heatmaps(img_path1, img_path2):

    try:
        img1_pil = Image.open(img_path1).convert("RGB")
        img2_pil = Image.open(img_path2).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to open or convert images: {str(e)}")
    

    img1_pil_resized = img1_pil.resize((224, 224))
    img2_pil_resized = img2_pil.resize((224, 224))
    
    inputs1 = clip_processor(images=img1_pil, return_tensors="pt").to(device)
    inputs2 = clip_processor(images=img2_pil, return_tensors="pt").to(device)

    inputs1["pixel_values"].requires_grad = True
    inputs2["pixel_values"].requires_grad = True

    features1_norm = clip_model.get_image_features(**inputs1)
    features1_norm_detached = features1_norm.detach()
    features1_norm_detached /= features1_norm_detached.norm(p=2, dim=-1, keepdim=True)
    
    features2_norm = clip_model.get_image_features(**inputs2)
    features2_norm_detached = features2_norm.detach()
    features2_norm_detached /= features2_norm_detached.norm(p=2, dim=-1, keepdim=True)

    target_for_img1 = SimilarityTarget(features2_norm_detached)
    target_for_img2 = SimilarityTarget(features1_norm_detached)

    grayscale_cam1 = cam(input_tensor=inputs1["pixel_values"], targets=[target_for_img1])[0, :]
    grayscale_cam2 = cam(input_tensor=inputs2["pixel_values"], targets=[target_for_img2])[0, :]
    
    img1_cv = np.array(img1_pil_resized)
    if img1_cv is None or img1_cv.size == 0:
        raise ValueError("Image 1 conversion to array failed")
    img1_cv_float = np.float32(img1_cv) / 255.0

    img2_cv = np.array(img2_pil_resized)
    if img2_cv is None or img2_cv.size == 0:
        raise ValueError("Image 2 conversion to array failed")
    img2_cv_float = np.float32(img2_cv) / 255.0

    cam_image1 = show_cam_on_image(img1_cv_float, grayscale_cam1, use_rgb=True)
    cam_image2 = show_cam_on_image(img2_cv_float, grayscale_cam2, use_rgb=True)

    _, buffer1 = cv2.imencode('.png', cv2.cvtColor(cam_image1, cv2.COLOR_RGB2BGR))
    sim1_b64 = base64.b64encode(buffer1).decode('utf-8')
    
    _, buffer2 = cv2.imencode('.png', cv2.cvtColor(cam_image2, cv2.COLOR_RGB2BGR))
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
        As an AI image analyst, I've determined two images have a semantic similarity score of {score*100:.1f}%.
        Briefly explain why they received this score.
        1.  **Similarities:** Note shared concepts, subjects, or composition.
        2.  **Differences:** Pinpoint conceptual differences that prevent a 100% score.
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
            content = await file1.read()
            buffer.write(content)
        with open(path2, "wb") as buffer:
            content = await file2.read()
            buffer.write(content)
        

        if not os.path.exists(path1) or os.path.getsize(path1) == 0:
            raise HTTPException(status_code=400, detail="Failed to save image 1 or file is empty")
        if not os.path.exists(path2) or os.path.getsize(path2) == 0:
            raise HTTPException(status_code=400, detail="Failed to save image 2 or file is empty")
        
        print(f"Processing images: {path1} ({os.path.getsize(path1)} bytes), {path2} ({os.path.getsize(path2)} bytes)")
        
        score = get_similarity_score(path1, path2)
        print(f"Similarity score calculated: {score}")
        
        heatmaps = generate_similarity_heatmaps(path1, path2) 
        print("Heatmaps generated successfully")
        
        explanation = get_llm_explanation(path1, path2, score)
        
        return JSONResponse(content={
            "similarity_percentage": round(float(score) * 100, 2),
            "ai_explanation": explanation,
            "heatmap_image1": f"data:image/png;base64,{heatmaps['similarity_img1']}",
            "heatmap_image2": f"data:image/png;base64,{heatmaps['similarity_img2']}"
        })

    except Exception as e:
        print(f"ERROR in compare_images: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    finally:
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)