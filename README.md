![15.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/uhmYm4HQqnQyMH4trg5V8.png)

# **Recycling-Net-11**

> **Recycling-Net-11** is an image classification model fine-tuned from **google/siglip2-base-patch16-224** using the **SiglipForImageClassification** architecture. The model classifies images into 11 categories related to recyclable materials, helping to automate and enhance waste sorting systems.

```py
Classification Report:
                   precision    recall  f1-score   support

        aluminium     0.9213    0.9145    0.9179       269
        batteries     0.9833    0.9933    0.9883       297
        cardboard     0.9660    0.9343    0.9499       274
disposable plates     0.9078    0.9744    0.9399       273
            glass     0.9621    0.9490    0.9555       294
     hard plastic     0.8675    0.7250    0.7899       280
            paper     0.8702    0.8941    0.8820       255
      paper towel     0.9333    0.9622    0.9475       291
      polystyrene     0.8188    0.8385    0.8285       291
    soft plastics     0.8425    0.8693    0.8557       283
    takeaway cups     0.9575    0.9767    0.9670       300

         accuracy                         0.9128      3107
        macro avg     0.9119    0.9119    0.9111      3107
     weighted avg     0.9127    0.9128    0.9119      3107
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/XW6fZXkQ2-Z5KhSjnxuQs.png)

The model categorizes images into the following classes:

- **0:** aluminium  
- **1:** batteries  
- **2:** cardboard  
- **3:** disposable plates  
- **4:** glass  
- **5:** hard plastic  
- **6:** paper  
- **7:** paper towel  
- **8:** polystyrene  
- **9:** soft plastics  
- **10:** takeaway cups  

---

![15.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/uhmYm4HQqnQyMH4trg5V8.png)

# **Run with Transformers 🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```

---

# **Intended Use**

**Recycling-Net-11** is ideal for:

- **Smart Waste Sorting:** Automating recycling processes in smart bins or factories.  
- **Environmental Awareness Tools:** Helping people learn how to sort waste correctly.  
- **Municipal Waste Management:** Classifying and analyzing urban waste data.  
- **Robotics:** Assisting robots in identifying and sorting materials.  
- **Education:** Teaching children and communities about recyclable materials.
