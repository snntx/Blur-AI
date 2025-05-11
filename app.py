import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
import tempfile
import os
from utils import (
    apply_blur, apply_square, delete_object, crop_to_object,
    blur_faces, blur_number_plates, prepare_download
)

# Initialize YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

def process_image(image, selected_object=None, tool=None, global_tool=None):
    """Process image with selected tools"""
    if global_tool == "Blur All Faces":
        # Get all person detections
        model = load_model()
        results = model(image)
        face_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if model.names[cls] == 'person':
                    face_boxes.append(box.xyxy[0].tolist())
        return blur_faces(image, face_boxes)
    
    elif global_tool == "Blur All Number Plates":
        # Get all vehicle detections
        model = load_model()
        results = model(image)
        plate_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if model.names[cls] in ['car', 'truck', 'bus', 'motorcycle']:
                    plate_boxes.append(box.xyxy[0].tolist())
        return blur_number_plates(image, plate_boxes)
    
    if selected_object and tool:
        box = selected_object['box']
        if tool == "Blur":
            return apply_blur(image, box)
        elif tool == "Square":
            return apply_square(image, box)
        elif tool == "Delete":
            return delete_object(image, box)
        elif tool == "Crop":
            return crop_to_object(image, box)
    
    return image

def main():
    st.set_page_config(
        page_title="BlurAI - AI Image Editor",
        page_icon="🎨",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .upload-area {
            border: 2px dashed #4CAF50;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🎨 BlurAI - AI Image Editor")
    
    # Initialize session state
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'detected_objects' not in st.session_state:
        st.session_state.detected_objects = []
    if 'selected_object' not in st.session_state:
        st.session_state.selected_object = None
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Canvas")
        
        # Create a styled upload area
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drag and drop an image here or click to upload", type=['png', 'jpg', 'jpeg'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.image = image
            
            # Display the image
            st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Object Detection")
        if st.session_state.image is not None:
            # Load YOLO model
            model = load_model()
            
            # Run object detection
            results = model(st.session_state.image)
            
            # Clear previous detections
            st.session_state.detected_objects = []
            
            # Display detected objects
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.5:  # Confidence threshold
                        st.session_state.detected_objects.append({
                            'class': model.names[cls],
                            'confidence': conf,
                            'box': box.xyxy[0].tolist()
                        })
            
            # Display object list with hover preview
            for obj in st.session_state.detected_objects:
                if st.radio(
                    f"{obj['class']} ({obj['confidence']:.2f})",
                    [obj['class']],
                    key=obj['class']
                ):
                    st.session_state.selected_object = obj
                    
                    # Show preview box on the image
                    preview_image = st.session_state.image.copy()
                    x1, y1, x2, y2 = map(int, obj['box'])
                    cv2.rectangle(
                        np.array(preview_image),
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )
                    st.image(preview_image, use_column_width=True)
            
            # Object tools
            if st.session_state.selected_object:
                st.subheader("Object Tools")
                tool = st.selectbox(
                    "Select tool",
                    ["Blur", "Square", "Delete", "Crop"]
                )
                
                if st.button("Apply Tool"):
                    st.session_state.image = process_image(
                        st.session_state.image,
                        st.session_state.selected_object,
                        tool
                    )
                    st.experimental_rerun()
            
            # Global tools
            st.subheader("Global Tools")
            global_tool = st.selectbox(
                "Select global tool",
                ["None", "Blur All Faces", "Blur All Number Plates"]
            )
            
            if st.button("Apply Global Tool"):
                st.session_state.image = process_image(
                    st.session_state.image,
                    global_tool=global_tool
                )
                st.experimental_rerun()
            
            # Download options
            st.subheader("Download")
            if st.button("Download Standard Resolution"):
                if st.session_state.image:
                    img_bytes = prepare_download(st.session_state.image)
                    st.download_button(
                        label="Click to download",
                        data=img_bytes,
                        file_name="edited_image.png",
                        mime="image/png"
                    )
            
            if st.button("Download High Resolution"):
                if st.session_state.image:
                    img_bytes = prepare_download(st.session_state.image, high_res=True)
                    st.download_button(
                        label="Click to download",
                        data=img_bytes,
                        file_name="edited_image_hires.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main() 