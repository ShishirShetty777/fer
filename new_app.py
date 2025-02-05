import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import timm
import torch.nn as nn
from torchvision import transforms
import os
import gdown

# Function to download Caffe files if they are not already available
def download_caffe_files():
    prototxt_url = 'https://drive.google.com/uc?id=1Akh9Qw1b_x9go3N6QXY1r2vbnyh8rJgJ'  # Replace with actual URL
    caffemodel_url = 'https://drive.google.com/uc?id=1OZ2scR5F7M0xQy73hhbxajXnl3W8th1x'  # Replace with actual URL

    # Download Caffe prototxt and model files if they are not already present
    if not os.path.exists('deploy.prototxt'):
        gdown.download(prototxt_url, 'deploy.prototxt', quiet=False)
    if not os.path.exists('res10_300x300_ssd_iter_140000_fp16.caffemodel'):
        gdown.download(caffemodel_url, 'res10_300x300_ssd_iter_140000_fp16.caffemodel', quiet=False)

# Download the Caffe files
download_caffe_files()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Correct model architecture matching the checkpoint
class EnhancedFERNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize backbone with original classifier structure
        self.backbone = timm.create_model('efficientnet_b1', pretrained=False)
        in_features = self.backbone.classifier.in_features
        
        # Rebuild classifier exactly as in original training
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 7)
        )

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model():
    model = EnhancedFERNet()
    
    # Replace with your actual Google Drive link to the model
    model_url = 'https://drive.google.com/uc?id=1oUKXqHOGntTZZ-5tgMSRDZllo4Yrt6Lk'
    output_path = 'combined_model_epoch39.pth'
    
    # Download model file from Google Drive using gdown
    gdown.download(model_url, output_path, quiet=False)

    state_dict = torch.load(output_path, map_location=device)
    
    # 1. Remove attention-related parameters from state_dict
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                         if not k.startswith('attention')}
    
    # 2. Load state dict with strict=False for verification
    load_result = model.load_state_dict(filtered_state_dict, strict=False)
    
    return model.eval().to(device)

model = load_model()

# Configuration
emotion_classes = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
emotion_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 165, 255), 
                 (255, 255, 0), (147, 20, 255), (128, 128, 128)]

# Initialize face detector
face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Critical: Preprocessing must match training exactly
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Must match original training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_faces(frame, confidence_threshold=0.7):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                               (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            faces.append(box.astype("int"))
    return faces

def predict_emotion(face_img):
    """Enhanced prediction with input validation"""
    try:
        # Convert to PIL Image and validate
        img = Image.fromarray(face_img).convert('RGB')
        
        # Verify image dimensions
        if img.size != (224, 224):
            img = img.resize((224, 224))
            
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            
        return probs.cpu().numpy()[0]
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return np.zeros(len(emotion_classes))

def process_image(image_array, det_conf=0.7, emo_conf=0.4):
    frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    faces = detect_faces(frame, det_conf)
    results = []
    
    for (startX, startY, endX, endY) in faces:
        try:
            # Extract and validate face ROI
            face_roi = frame[startY:endY, startX:endX]
            if face_roi.size == 0 or min(face_roi.shape[:2]) < 50:
                continue
                
            # Convert to RGB and predict
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            probabilities = predict_emotion(face_rgb)
            max_prob = probabilities.max()
            
            if max_prob < emo_conf:
                continue
                
            emotion_idx = np.argmax(probabilities)
            emotion = emotion_classes[emotion_idx]
            
            # Draw results
            color = emotion_colors[emotion_idx]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, f"{emotion} ({max_prob:.2f})", 
                       (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            results.append({
                "emotion": emotion,
                "confidence": float(max_prob),
                "position": (startX, startY, endX-startX, endY-startY),
                "probabilities": probabilities.tolist()
            })
        except Exception as e:
            continue
            
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), results

# Streamlit UI with verification tools
st.title("Facial Emotion Recognition")
st.write("Upload an image to analyze facial expressions")

with st.sidebar:
    st.header("Settings")
    det_conf = st.slider("Face Detection Confidence", 0.1, 1.0, 0.7, 0.05)
    emo_conf = st.slider("Emotion Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
    debug_mode = st.checkbox("Enable Debug Mode")

# Model verification section
if debug_mode:
    st.subheader("Model Verification")
    
    # Test with random input
    test_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    st.write("Test output logits:", test_output.cpu().numpy())
    
    # Check class distribution
    st.write("Test probabilities:", torch.softmax(test_output, dim=1).cpu().numpy())

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
    
    processed_image, faces = process_image(image_array, det_conf, emo_conf)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(processed_image, caption="Analysis Results", use_column_width=True)
    
    if faces:
        st.success(f"Detected {len(faces)} faces")
        for i, face in enumerate(faces):
            with st.expander(f"Face {i+1}: {face['emotion'].capitalize()} ({face['confidence']:.2f})"):
                st.write(f"**Position:** X={face['position'][0]}, Y={face['position'][1]}")
                st.write(f"**Size:** {face['position'][2]}x{face['position'][3]} pixels")
                st.progress(face['confidence'])
                st.bar_chart({e: p for e, p in zip(emotion_classes, face['probabilities'])})
    else:
        st.warning("No faces detected meeting confidence thresholds")
