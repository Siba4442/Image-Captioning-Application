import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

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

# Function to generate captions
def predict_step(images, num_captions):
    """Generate captions for a list of images."""
    processed_images = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        processed_images.append(image)

    pixel_values = feature_extractor(images=processed_images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    gen_kwargs = {"max_length": 16, "num_beams": 4, "num_return_sequences": num_captions}
    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [caption.strip() for caption in preds]
    return preds

# Streamlit app
def main():
    # Center the title and description
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Image Captioning Application</h1>
            <p>Upload an image to generate captions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Option to select number of captions
    num_captions = st.slider("Select number of captions to generate", min_value=1, max_value=5, value=1)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False)

    if uploaded_file:
        try:
            # Ensure that the file is an image
            image = Image.open(uploaded_file)
            image = image.convert("RGB")  # Ensure it's in RGB mode

            # Generate captions
            captions = predict_step([image], num_captions)

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Display the captions
            st.subheader("Generated Captions:")
            for i, caption in enumerate(captions, start=1):
                st.success(f"{i}. {caption}")

        except Exception as e:
            st.error(f"Error: {e}. Please upload a valid image.")

if __name__ == "__main__":
    main()
