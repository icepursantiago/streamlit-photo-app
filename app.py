import streamlit as st
from PIL import Image
import io

st.title("Phone Photo Uploader")

img_file = st.camera_input("Take a photo")  # works well on mobile
# Alternatively (if camera_input isn't available in your setup):
# img_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

if img_file:
    image = Image.open(img_file)
    st.image(image, caption="Captured image", use_container_width=True)

    # Example “processing”: convert to grayscale (lightweight demo)
    gray = image.convert("L")
    st.image(gray, caption="Grayscale result", use_container_width=True)

    # Provide a download button for processed image
    buf = io.BytesIO()
    gray.save(buf, format="PNG")
    st.download_button(
        "Download processed image",
        data=buf.getvalue(),
        file_name="processed.png",
        mime="image/png",
    )
