import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request

# --- Model URLs ---
CFG_URL = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg'
WEIGHTS_URL = 'https://pjreddie.com/media/files/yolov3-tiny.weights'
NAMES_URL = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
CFG_PATH = 'yolov3-tiny.cfg'
WEIGHTS_PATH = 'yolov3-tiny.weights'
NAMES_PATH = 'coco.names'

# --- Download Model Files if Needed ---
def download_file(url, path):
    if not os.path.exists(path):
        with urllib.request.urlopen(url) as response, open(path, 'wb') as out_file:
            out_file.write(response.read())

def ensure_model_files():
    download_file(CFG_URL, CFG_PATH)
    download_file(WEIGHTS_URL, WEIGHTS_PATH)
    download_file(NAMES_URL, NAMES_PATH)

# --- Load Class Names ---
def load_class_names():
    with open(NAMES_PATH, 'r') as f:
        return [line.strip() for line in f.readlines()]

# --- Load Model ---
@st.cache_resource
def load_model():
    ensure_model_files()
    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    classes = load_class_names()
    return net, classes

# --- Detect Objects ---
def detect_objects(image, net, classes, conf_threshold=0.4, nms_threshold=0.3):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    ln = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(ln)
    boxes, confidences, classIDs = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            results.append({
                'class': classes[classIDs[i]],
                'confidence': confidences[i],
                'box': [x, y, x + w, y + h]
            })
    return results

# --- FontAwesome Icon Map ---
FA_ICONS = {
    'person': 'fa-user', 'car': 'fa-car', 'bus': 'fa-bus', 'truck': 'fa-truck',
    'bicycle': 'fa-bicycle', 'motorbike': 'fa-motorcycle', 'dog': 'fa-dog', 'cat': 'fa-cat',
    'bird': 'fa-dove', 'boat': 'fa-ship', 'aeroplane': 'fa-plane', 'train': 'fa-train',
    'tvmonitor': 'fa-tv', 'bottle': 'fa-wine-bottle', 'chair': 'fa-chair', 'sofa': 'fa-couch',
    'pottedplant': 'fa-seedling', 'sheep': 'fa-hippo', 'cow': 'fa-cow', 'horse': 'fa-horse',
    'diningtable': 'fa-utensils', 'background': 'fa-cube'
}

def fa_icon(label):
    return f'<i class="fa {FA_ICONS.get(label, "fa-cube")}"></i>'

# --- UI ---
def main():
    st.set_page_config(page_title="BlurAI - Image Editor", layout="wide")
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    body, .main, .block-container { background: #18191A !important; color: #fff; }
    .topbar { display: flex; align-items: center; padding: 0.5rem 1.5rem; background: #232323; border-bottom: 1px solid #333; }
    .topbar-title { font-size: 2rem; font-weight: bold; letter-spacing: 1px; margin-right: 2rem; }
    .tool-btn { background: #232323; color: #fff; border: none; border-radius: 6px; padding: 0.6rem 1.2rem; margin-right: 0.5rem; font-size: 1rem; display: inline-flex; align-items: center; gap: 0.5rem; transition: background 0.2s; }
    .tool-btn:hover { background: #333; }
    .sidebar-objects { background: #232323; border-left: 1px solid #333; min-width: 270px; height: 100vh; padding: 1.5rem 1rem; position: fixed; right: 0; top: 0; z-index: 10; }
    .object-card { background: #18191A; border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.7rem; border: 1px solid #333; cursor: pointer; transition: border 0.2s; }
    .object-card.selected, .object-card:hover { border: 1.5px solid #4CAF50; background: #232323; }
    .object-label { font-size: 1rem; font-weight: 500; }
    .canvas-area { display: flex; align-items: center; justify-content: center; height: 75vh; background: #18191A; }
    .upload-placeholder { background: #232323; border: 2px dashed #4CAF50; border-radius: 12px; padding: 3rem 4rem; text-align: center; color: #fff; font-size: 1.2rem; }
    .download-btn { margin-top: 1.5rem; background: #4CAF50; color: #fff; border: none; border-radius: 6px; padding: 0.7rem 1.5rem; font-size: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    # --- Top Bar ---
    st.markdown('<div class="topbar">'
        '<span class="topbar-title">BlurAI</span>'
        '<button class="tool-btn"><i class="fa fa-eye-slash"></i> Blur</button>'
        '<button class="tool-btn"><i class="fa fa-trash"></i> Delete</button>'
        '<button class="tool-btn"><i class="fa fa-user"></i> Blur Faces</button>'
        '<button class="tool-btn"><i class="fa fa-car"></i> Blur Plates</button>'
        '<button class="tool-btn"><i class="fa fa-ad"></i> Blur Ads</button>'
        '</div>', unsafe_allow_html=True)

    # --- Main Layout ---
    col_canvas, _ = st.columns([5, 2], gap="large")
    with col_canvas:
        st.write("")  # Spacer
        if 'image' not in st.session_state:
            st.session_state.image = None
        uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.image = image
        if st.session_state.image is not None:
            st.markdown('<div class="canvas-area">', unsafe_allow_html=True)
            st.image(st.session_state.image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="canvas-area"><div class="upload-placeholder">'
                        '<div style="font-size:2.5rem;"><i class="fa fa-cloud-upload-alt"></i></div>'
                        'Drag and drop your image here<br>or<br><b>Click to upload</b>'
                        '</div></div>', unsafe_allow_html=True)
        if st.session_state.image is not None:
            img_bytes = st.session_state.image.copy()
            st.download_button("⬇️ Download", img_bytes, file_name="edited_image.png", mime="image/png", key="download_btn", help="Download your edited image", use_container_width=True)

    # --- Sidebar: Detected Objects ---
    st.markdown('<div class="sidebar-objects">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.2rem;font-weight:bold;margin-bottom:1.2rem;">Detected Objects</div>', unsafe_allow_html=True)
    detected_objects = []
    if st.session_state.image is not None:
        net, classes = load_model()
        np_img = np.array(st.session_state.image.convert('RGB'))
        objects = detect_objects(np_img, net, classes)
        for idx, obj in enumerate(objects):
            icon = fa_icon(obj['class'])
            selected = st.session_state.get('selected_object_idx', -1) == idx
            if st.button(f"{icon} <span class='object-label'>{obj['class'].capitalize()} ({obj['confidence']:.2f})</span>", key=f"obj_{idx}", help="Select object", use_container_width=True):
                st.session_state.selected_object_idx = idx
                st.session_state.selected_object = obj
            st.markdown(f'<div class="object-card{ ".selected" if selected else "" }">{icon} <span class="object-label">{obj["class"].capitalize()} ({obj["confidence"]:.2f})</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#888;">No objects detected yet.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 