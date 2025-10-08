import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Load the trained model
model = YOLO("result/bone_fracture_yolov8_optimized/weights/best.pt")  # Adjust path if needed

# App title
st.header('Bone Fracture Detection App')
st.subheader('Powered by YOLOv8')
st.write('Upload an X-ray image to detect bone fractures.')

# File uploader
uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])

# Class selection and confidence slider
object_names = list(model.names.values())
selected_objects = st.multiselect('Choose fracture types to detect', object_names, default=object_names)
min_confidence = st.slider('Confidence score', 0.0, 1.0, 0.5)

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    img_cv2 = np.array(image)  # Convert to OpenCV format
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Perform prediction
    with st.spinner('Detecting...'):
        results = model(img_cv2)

    # Process results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, score, cls in zip(boxes, scores, classes):
            if model.names[int(cls)] in selected_objects and score > min_confidence:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{model.names[int(cls)]} {score:.2f}'
                cv2.putText(img_cv2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the result
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption='Detected Image', use_column_width=True)