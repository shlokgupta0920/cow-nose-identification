import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import threading

MODEL_NAME = "facebook/dinov2-small"
device = torch.device("cpu")

processor = None
model = None
_model_init_lock = threading.Lock()
_inference_lock = threading.Lock()


def load_ai_model():
    global processor, model

    if model is None:
        with _model_init_lock:
            if model is None:
                processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
                model = AutoModel.from_pretrained(MODEL_NAME)
                model.to(device)
                model.eval()

    return model


def extract_features(image):
    load_ai_model()

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with _inference_lock:
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0]
            features = F.normalize(features, p=2, dim=1)

    return features


def compare_images(image1, image2):
    feature1 = extract_features(image1)
    feature2 = extract_features(image2)

    similarity = F.cosine_similarity(feature1, feature2)
    return float(similarity.item())
