import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Recycling-Net-11"  # Update with your actual Hugging Face model path
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    0: "aluminium",
    1: "batteries",
    2: "cardboard",
    3: "disposable plates",
    4: "glass",
    5: "hard plastic",
    6: "paper",
    7: "paper towel",
    8: "polystyrene",
    9: "soft plastics",
    10: "takeaway cups"
}

def classify_recyclable_material(image):
    """Predicts the type of recyclable material in the image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {id2label[i]: round(probs[i], 3) for i in range(len(probs))}
    return predictions

# Gradio interface
iface = gr.Interface(
    fn=classify_recyclable_material,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Recyclable Material Prediction Scores"),
    title="Recycling-Net-11",
    description="Upload an image of a waste item to identify its recyclable material type."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
