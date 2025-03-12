import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import faiss
from skimage.feature import hog
from PIL import Image
import streamlit as st

# Load VGG19 Model
base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Initialize session state for index and image paths
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.image_paths = []


# Feature Extraction
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    deep_features = model.predict(img).flatten()

    gray_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (128, 128))
    texture_features = hog(
        gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True
    )

    return np.concatenate((deep_features, texture_features))


# Load Dataset
def load_dataset(dataset_folder):
    image_paths = []
    feature_vectors = []

    for file_name in os.listdir(dataset_folder):
        if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
            path = os.path.join(dataset_folder, file_name)
            features = extract_features(path)
            image_paths.append(path)
            feature_vectors.append(features)

    feature_vectors = np.array(feature_vectors, dtype="float32")

    # Build FAISS Index
    dimension = feature_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(feature_vectors)

    # Store in session state
    st.session_state.index = index
    st.session_state.image_paths = image_paths

    st.success(f"✅ Loaded {len(image_paths)} images successfully!")


# Search Image
def search_image(query_path):
    if st.session_state.index is None:
        st.warning("⚠️ Load the dataset first!")
        return None, None

    query_features = extract_features(query_path)
    D, I = st.session_state.index.search(np.array([query_features]), k=1)

    matched_path = st.session_state.image_paths[I[0][0]]
    similarity_score = D[0][0] * 100
    return matched_path, similarity_score


# Streamlit UI
def main():
    st.set_page_config(page_title="🔎 Similar Image Search", layout="centered")

    st.title("🔎 Similar Image Search")
    st.write("Upload a dataset and search for similar images.")

    # Load Dataset Button
    dataset_folder = st.text_input("📂 Enter path to dataset folder")
    if st.button("Load Dataset"):
        if dataset_folder and os.path.exists(dataset_folder):
            load_dataset(dataset_folder)
        else:
            st.error("❌ Invalid folder path!")

    # Upload Query Image
    query_image = st.file_uploader("📸 Upload an Image to Search", type=["jpg", "jpeg", "png", "webp"])
    
    if query_image and st.button("🔍 Search"):
        # Save query image temporarily
        query_path = "temp_query." + query_image.name.split('.')[-1]
        with open(query_path, "wb") as f:
            f.write(query_image.getbuffer())

        # Search for similar image
        matched_path, similarity_score = search_image(query_path)

        if matched_path:
            col1, col2 = st.columns(2)

            with col1:
                st.image(query_path, caption="Query Image", width=200)

            with col2:
                st.image(matched_path, caption=f"Matched Image ({similarity_score:.2f}%)", width=200)
            
            st.progress(similarity_score / 100)
            st.write(f"**🔍 Similarity:** {similarity_score:.2f}%")

        else:
            st.warning("⚠️ No matching image found.")

if __name__ == "__main__":
    main()
