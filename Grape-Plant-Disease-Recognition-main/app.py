# Improved Streamlit app for Grape Plant Disease Detection (with camera input + login)

import streamlit as st
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import time

st.set_page_config(page_title="AgriCure", layout="centered", page_icon="🍇")

# --------------------------------------------------
# SAFE RERUN FUNCTION (fix experimental_rerun error)
# --------------------------------------------------
def safe_rerun():
    try:

        
        st.experimental_rerun()
    except Exception:
        try:
            st.experimental_set_query_params(_refresh=int(time.time()))
        except:
            st.warning("Please refresh the page manually.")

# --------------------------------------------------
# SIMPLE LOGIN SYSTEM
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

def do_login(user, pwd):
    valid_users = {"team068": "123456", "admin": "admin123"}
    return user in valid_users and valid_users[user] == pwd

def do_logout():
    st.session_state.logged_in = False
    st.session_state.username = None

# -----------------------
# LOGIN PAGE RENDER
# -----------------------
if not st.session_state.logged_in:

    st.markdown("""
        <h1 style='text-align:center; color:#FFD700;'>🔐 AgriCure Login</h1>
        <p style='text-align:center; color:#ccc;'>Please login to continue</p>
    """, unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

        if login_btn:
            if do_login(username.strip(), password.strip()):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                safe_rerun()
            else:
                st.error("Invalid username or password")

    st.stop()

# If logged in → show logout button
st.sidebar.success(f"Logged in as: {st.session_state.username}")
if st.sidebar.button("Logout"):
    do_logout()
    safe_rerun()

# --------------------------------------------------
# MAIN PAGE STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main {background-color: #0e1117; color: #d6e0f0}
    .stApp { background-color: #0e1117; }
    .card {background:#0b1220; padding:18px; border-radius:12px}
    ul.custom {padding-left:16px; margin-top:6px}
    ul.custom li {margin-bottom:8px}
    </style>
    """,
    unsafe_allow_html=True,
)

LOGO_PATH = "assets/logo.png"

# --------------------------------------------------
# MODEL + UTILS
# --------------------------------------------------
def find_model(models_dir="models"):
    if not os.path.exists(models_dir):
        return None
    for fname in os.listdir(models_dir):
        if fname.endswith(".h5"):
            return os.path.join(models_dir, fname)
    return None

def load_class_names(path="labels.txt"):
    if os.path.exists(path):
        with open(path) as f:
            return [x.strip() for x in f.readlines() if x.strip()]
    return ["Black Rot", "ESCA", "Leaf Blight", "Healthy"]

IMG_SIZE = (224, 224)

REMEDIES = {
    "Black Rot": {
        "description": "Fungal disease causing black lesions on leaves.",
        "actions": [
            "Remove affected leaves.",
            "Improve ventilation.",
            "Apply fungicides as recommended.",
        ],
        "severity": "High",
    },
    "ESCA": {
        "description": "Trunk disease causing decay & leaf discoloration.",
        "actions": [
            "Remove severely infected vines.",
            "Maintain healthy irrigation.",
        ],
        "severity": "High",
    },
    "Leaf Blight": {
        "description": "Causes brown lesions & defoliation.",
        "actions": [
            "Destroy infected leaves.",
            "Apply proper fungicides.",
        ],
        "severity": "Medium",
    },
    "Healthy": {
        "description": "Leaf shows no visible signs of disease.",
        "actions": [
            "Continue routine care.",
            "Monitor regularly.",
        ],
        "severity": "None",
    },
}

# Load model and names
model_path = find_model()
class_names = load_class_names()

@st.cache_resource
def get_model(path):
    if not path:
        return None
    return load_model(path)

model = get_model(model_path)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
col1, col2 = st.columns([1, 3])

with col1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=180)
    else:
        st.warning("Logo missing: assets/logo.png")

with col2:
    st.markdown("""
        <h1 style='text-align:center; color:#FFD700; font-size:48px;'>
            AgriCure – Grape Vines Disease Detection
        </h1>
        <p style='text-align:center; color:#ccc; font-size:20px;'>
            Upload a grape leaf or use your camera to diagnose diseases.
        </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# UPLOADER + CAMERA + INFO SECTION
# --------------------------------------------------
left, right = st.columns([1, 1.2])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
    st.write("— or —")
    cam_img = st.camera_input("Take a photo")
    st.markdown("</div>", unsafe_allow_html=True)

    input_file = cam_img if cam_img else uploaded
    input_label = "Camera" if cam_img else ("Upload" if uploaded else None)

    if input_file is None:
        st.info("Upload an image or take a photo.")

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model & Info")

    if not model:
        st.warning("Model not found in /models folder.")
    else:
        st.success(f"Model loaded: {os.path.basename(model_path)}")
        st.write(f"Classes: {', '.join(class_names)}")

    st.markdown("---")
    st.subheader("How to Use")
    st.markdown("""
        <ul class="custom">
            <li>Upload a leaf image or take a clear photo.</li>
            <li>Ensure the leaf is well-lit and centered.</li>
            <li>Click <b>Predict</b> to run disease detection.</li>
            <li>Scroll down to view disease details & remedies.</li>
        </ul>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# PREDICTION BLOCK
# --------------------------------------------------
def read_image(file):
    try:
        data = file.read()
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None

if input_file and model:
    img = read_image(input_file)

    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"{input_label} image", use_column_width=True)

        img_resized = cv2.resize(img, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(img_rgb.astype("float32") / 255.0, axis=0)

        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                preds = model.predict(inp)
                probs = preds[0]
                idx = int(np.argmax(probs))
                label = class_names[idx]
                confidence = probs[idx]

            st.success(f"Prediction: **{label}** ({confidence*100:.2f}%)")

            df = pd.DataFrame({
                "class": class_names,
                "probability": (probs * 100).round(2)
            }).sort_values("probability", ascending=False)

            st.write(df.reset_index(drop=True))
            st.bar_chart(df.set_index("class"))

            st.markdown("---")
            st.subheader("Recommended Actions & Remedies")

            info = REMEDIES.get(label, {})
            st.write(f"**Description:** {info.get('description', 'N/A')}")
            st.write(f"**Severity:** {info.get('severity', 'N/A')}")
            st.write("**Actions:**")
            for a in info.get("actions", []):
                st.write(f"- {a}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    "<p style='text-align:center; color:#9aa6b2;'>Built by Team 067 with ❤️</p>",
    unsafe_allow_html=True
)
