import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

# -------------------------------
# 🎨 Giao diện
# -------------------------------
st.set_page_config(page_title="Bone Fracture Detection", page_icon="🦴", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
        font-family: "Segoe UI", sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    section[data-testid="stFileUploader"] {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 20px;
        background-color: #ecf6fd;
    }
    img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 🔍 Tiêu đề ứng dụng
# -------------------------------
st.markdown('<div class="title">🦴 Bone Fracture Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by YOLOv8 - Upload an X-ray to detect fractures</div>', unsafe_allow_html=True)

# -------------------------------
# ⚙️ Load mô hình
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("result/bone_fracture_yolov8_optimized/weights/best.pt")

model = load_model()

# -------------------------------
# 🧩 Sidebar - Cấu hình
# -------------------------------
st.sidebar.header("⚙️ Detection Settings")
object_names = list(model.names.values())
selected_objects = st.sidebar.multiselect('Select fracture types', object_names, default=object_names)
min_confidence = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5)

# -------------------------------
# 📤 Upload ảnh
# -------------------------------
uploaded_file = st.file_uploader("📸 Upload an X-ray image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_cv2 = np.array(image)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

    with st.spinner('🔍 Detecting fractures...'):
        results = model(img_cv2)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            if model.names[int(cls)] in selected_objects and score > min_confidence:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (46, 204, 113), 3)
                label = f'{model.names[int(cls)]} {score:.2f}'
                cv2.putText(img_cv2, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (52, 152, 219), 2)

    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # Hiển thị ảnh kết quả
    st.image(img_rgb, caption='🩻 Detection Result', use_column_width=True)

    # 💾 Nút lưu ảnh
    buffered = BytesIO()
    result_image = Image.fromarray(img_rgb)
    result_image.save(buffered, format="PNG")
    st.download_button(
        label="💾 Download Result Image",
        data=buffered.getvalue(),
        file_name="fracture_detection_result.png",
        mime="image/png"
    )

else:
    st.info("📂 Please upload an X-ray image to begin detection.")
