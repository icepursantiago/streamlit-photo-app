import io
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf


MODEL_PATH = Path("malaria_model.tflite")
INPUT_SIZE = (64, 64)
CLASS_NAMES = ["Parasitized", "Uninfected"]


def load_interpreter(model_path: Path) -> tf.lite.Interpreter:
	if not model_path.exists():
		raise FileNotFoundError(
			"Model file not found. Run the export cell to create malaria_model.tflite."
		)
	interpreter = tf.lite.Interpreter(model_path=str(model_path))
	interpreter.allocate_tensors()
	return interpreter


def preprocess_image(image: Image.Image) -> np.ndarray:
	image = image.convert("RGB")
	image = image.resize(INPUT_SIZE)
	array = np.array(image, dtype=np.float32) / 255.0
	return array[np.newaxis, ...]


def run_inference(interpreter: tf.lite.Interpreter, input_tensor: np.ndarray) -> np.ndarray:
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	interpreter.set_tensor(input_details[0]["index"], input_tensor)
	interpreter.invoke()
	return interpreter.get_tensor(output_details[0]["index"])  # shape: (1, 2)


st.set_page_config(page_title="Malaria Classifier", page_icon="🩸", layout="centered")
st.title("Malaria Cell Classifier")
st.write(
	"Take a photo from your phone, upload it here, and the model runs on your PC."
)

with st.sidebar:
	st.header("Model")
	st.write(f"Model file: {MODEL_PATH}")

try:
	interpreter = load_interpreter(MODEL_PATH)
except FileNotFoundError as exc:
	st.error(str(exc))
	st.stop()

photo = st.camera_input("Take a photo")
upload = st.file_uploader("...or upload an image", type=["png", "jpg", "jpeg"])

image_data = None
if photo is not None:
	image_data = photo.getvalue()
elif upload is not None:
	image_data = upload.getvalue()

if image_data:
	image = Image.open(io.BytesIO(image_data))
	st.image(image, caption="Input image", use_container_width=True)

	input_tensor = preprocess_image(image)
	probabilities = run_inference(interpreter, input_tensor)[0]
	predicted_index = int(np.argmax(probabilities))

	st.subheader("Prediction")
	st.write(f"Class: **{CLASS_NAMES[predicted_index]}**")
	st.write(f"Confidence: {probabilities[predicted_index]:.2%}")

	st.caption(
		"Note: model expects 64x64 RGB images and outputs two probabilities."
	)
