import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Image Processing App", layout="centered")

st.title("🖼️ Image Processing and Pattern Analysis App")
st.write("Apply filters and image operations interactively using Streamlit!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def contrast_stretch(img):
    a, b = 0, 255
    c, d = np.min(img), np.max(img)
    stretched = ((img - c) * ((b - a)/(d - c)) + a).astype(np.uint8)
    return stretched

def add_noise(img, noise_type):
    row, col, ch = img.shape
    if noise_type == "Gaussian":
        mean = 0
        var = 0.01
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, img.shape).reshape(img.shape)
        noisy = np.clip(img / 255 + gauss, 0, 1) * 255
        return noisy.astype(np.uint8)

    elif noise_type == "Salt & Pepper":
        s_vs_p = 0.5
        amount = 0.02
        out = np.copy(img)
        # Salt
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        out[tuple(coords)] = 255
        # Pepper
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        out[tuple(coords)] = 0
        return out

    elif noise_type == "Erlang":
        shape = img.shape
        lam = 5
        erlang_noise = np.random.gamma(lam, 1.0, shape)
        noisy = np.clip(img + erlang_noise, 0, 255)
        return noisy.astype(np.uint8)

    elif noise_type == "Rayleigh":
        shape = img.shape
        scale = 10
        rayleigh_noise = np.random.rayleigh(scale, shape)
        noisy = np.clip(img + rayleigh_noise, 0, 255)
        return noisy.astype(np.uint8)

def apply_fft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum.astype(np.uint8)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Original Image", use_column_width=True)

    operation = st.selectbox("Choose an Operation", [
        "Grayscale",
        "Gaussian Blur",
        "Median Blur",
        "Canny Edge Detection",
        "Histogram Equalization",
        "Color Space Conversion (HSV/LAB)",
        "Contrast Stretching",
        "FFT Transform (Magnitude)",
        "Low-pass Filter",
        "High-pass Filter",
        "Add Noise"
    ])

    processed_image = None

    if operation == "Grayscale":
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif operation == "Gaussian Blur":
        k = st.slider("Kernel Size", 1, 15, 3, step=2)
        processed_image = cv2.GaussianBlur(image, (k, k), 0)

    elif operation == "Median Blur":
        k = st.slider("Kernel Size", 1, 15, 3, step=2)
        processed_image = cv2.medianBlur(image, k)

    elif operation == "Canny Edge Detection":
        t1 = st.slider("Threshold1", 0, 255, 100)
        t2 = st.slider("Threshold2", 0, 255, 200)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.Canny(gray, t1, t2)

    elif operation == "Histogram Equalization":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.equalizeHist(gray)

    elif operation == "Color Space Conversion (HSV/LAB)":
        mode = st.radio("Convert to:", ["HSV", "LAB"])
        if mode == "HSV":
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    elif operation == "Contrast Stretching":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = contrast_stretch(gray)

    elif operation == "FFT Transform (Magnitude)":
        processed_image = apply_fft(image)

    elif operation == "Low-pass Filter":
        k = st.slider("Kernel Size", 1, 15, 3, step=2)
        kernel = np.ones((k, k), np.float32) / (k * k)
        processed_image = cv2.filter2D(image, -1, kernel)

    elif operation == "High-pass Filter":
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
        processed_image = cv2.filter2D(image, -1, kernel)

    elif operation == "Add Noise":
        noise_type = st.selectbox("Choose noise type", ["Gaussian", "Salt & Pepper", "Erlang", "Rayleigh"])
        processed_image = add_noise(image, noise_type)

    if processed_image is not None:
        st.subheader("Processed Image")
        if len(processed_image.shape) == 2:
            st.image(processed_image, use_column_width=True, clamp=True, channels="GRAY")
        else:
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)

        _, buffer = cv2.imencode('.png', processed_image)
        st.download_button(
            label="Download Processed Image",
            data=buffer.tobytes(),
            file_name="processed_image.png",
            mime="image/png"
        )
