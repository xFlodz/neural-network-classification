import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# URL до API
API_URL = "https://chicken-cow-horse-sheep-classification.onrender.com/predict/"

# Имена классов
CLASS_NAMES = {
    "0": "chicken",
    "1": "cow",
    "2": "dog",
    "3": "elephant"
}

st.title("Загрузи или нарисуй изображение для классификации животного")

tab = st.radio("Выбери, что ты хочешь", ["Загрузить изображение", "Нарисовать животное"])

image = None

if tab == "Загрузить изображение":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif tab == "Нарисовать животное":
    st.write("Нарисуйте животное белым цветом на черном фоне:")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=256,
        height=256,
        drawing_mode="freedraw",
        key="canvas_draw",
        update_streamlit=True
    )

    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data).astype("uint8")).convert("RGB")

if image:
    st.image(image, caption="Входное изображение", use_container_width=True)

    if st.button("Классифицировать"):
        img_resized = image.resize((64, 64))
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        files = {"file": ("image.png", img_bytes, "image/png")}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            class_id = result.get("predicted_class", "unknown")
            class_name = CLASS_NAMES.get(class_id, f"Unknown class: {class_id}")

            st.subheader("Вероятнее всего на изображении:")
            st.write(class_name)

            raw_probs = result.get("probabilities", {})
            readable_probs = {CLASS_NAMES.get(k, k): v for k, v in raw_probs.items()}

            st.subheader("Однако, если я ошибся - посмотри вероятности того, кто может быть на изображении")
            fig, ax = plt.subplots()
            ax.bar(readable_probs.keys(), readable_probs.values(), color="skyblue")
            ax.set_ylabel("Вероятность")
            ax.set_ylim([0, 1])
            st.pyplot(fig)
        else:
            st.error("Ошибка при обращении к API:")
            st.text(response.text)
