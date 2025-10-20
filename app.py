import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os

model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def preprocess_image(img_path):
    """Loads an image from a file path and prepares it for the model."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Error: Image file not found at '{img_path}'")
        
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    img_array = np.expand_dims(img, axis=0)
    return preprocess_input(img_array)

def extract_features(img_array):
    """Extracts a feature vector from a preprocessed image array."""
    features = model.predict(img_array)
    return features.flatten() # .flatten() is still needed to make it a 1D vector

def get_similarity(img_path1, img_path2):
    """Calculates the cosine similarity between two images."""
    img1_processed = preprocess_image(img_path1)
    img2_processed = preprocess_image(img_path2)
    
    print("Extracting features from Image 1...")
    features1 = extract_features(img1_processed).reshape(1, -1)
    print("Extracting features from Image 2...")
    features2 = extract_features(img2_processed).reshape(1, -1)
    
    similarity_score = cosine_similarity(features1, features2)[0][0]
    return similarity_score

if __name__ == "__main__":
    image_path_1 = 'image1.png'
    image_path_2 = 'image2.png'
    
    try:
        score = get_similarity(image_path_1, image_path_2)
        
        print("\n--- Similarity Result ---")
        print(f"The two images score is {score:.2f}.")
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")