import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import io

# Load model, tokenizer, and processor only once
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, feature_extractor, tokenizer, device

model, feature_extractor, tokenizer, device = load_model()

# Set generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "num_return_sequences": 1}  # Only one caption

def predict_step(images):
    """Generate captions for a list of images."""
    processed_images = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        processed_images.append(image)

    pixel_values = feature_extractor(images=processed_images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)  # Only one caption

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [caption.strip() for caption in preds]
    return preds

# Streamlit app
def main():
    st.title("Image Captioning Application")
    st.write("Upload an image to generate a caption.")

    # Center the page content
    st.markdown(
        """
        <style>
            .center {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                flex-direction: column;
                text-align: center;
            }
            .caption-box {
                background-color: #32CD32; /* Glowing green */
                color: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 0 20px 5px #32CD32;
                font-size: 18px;
                font-weight: bold;
                margin-top: 20px;
                width: 80%;
                word-wrap: break-word;
            }
            .title {
                color: #4B0082; /* Indigo */
                font-size: 30px;
                font-weight: bold;
            }
            .description {
                color: #000080; /* Navy Blue */
                font-size: 18px;
                margin-top: 10px;
            }
        </style>
        """, unsafe_allow_html=True)

    # File uploader
    uploaded_files = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False)

    if uploaded_files:
        try:
            # Ensure that the file is an image
            image = Image.open(uploaded_files)
            image = image.convert("RGB")  # Ensure it's in RGB mode

            # Clear previous images and captions when a new file is uploaded
            if "image_placeholder" in st.session_state:
                del st.session_state["image_placeholder"]

            if "caption_placeholder" in st.session_state:
                del st.session_state["caption_placeholder"]

            # Store new image and caption in session state
            st.session_state["image_placeholder"] = image

            # Generate captions
            captions = predict_step([image])
            st.session_state["caption_placeholder"] = captions[0]

            # Display the uploaded image
            st.markdown('<div class="center">', unsafe_allow_html=True)
            st.image(st.session_state["image_placeholder"], use_container_width=True)

            # Display the caption in a glowing green box
            caption_html = f"""
            <div class="caption-box">
                {st.session_state["caption_placeholder"]}
            </div>
            """
            st.markdown(caption_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}. Please upload a valid image.")

if __name__ == "__main__":
    main()
