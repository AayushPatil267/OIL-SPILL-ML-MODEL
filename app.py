import base64

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile


app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def load_models():
    global classifier, segmentation_model

    classifier = models.resnet18(weights=None)
    classifier.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, 1)
    classifier.load_state_dict(torch.load("classifier/oil_spill_classifier.pth", map_location=device))
    classifier.to(device)
    classifier.eval()

    segmentation_model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1, activation=None
    )
    segmentation_model.load_state_dict(torch.load("segmentation/oil_spill_segmentation.pth", map_location=device))
    segmentation_model.to(device)
    segmentation_model.eval()
    print("Both models loaded successfully.")


@app.get("/")
def root():
    return {"status": "Oil Spill Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"error": "Unable to decode image as grayscale."}

    original_height, original_width = image.shape[:2]

    try:
        classifier_image = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
        classifier_tensor = torch.from_numpy(classifier_image).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            confidence = torch.sigmoid(classifier(classifier_tensor)).item()
    except Exception as error:
        return {"error": str(error)}

    if confidence < 0.5:
        return {"spill_detected": False, "confidence": round(confidence, 4)}

    try:
        segmentation_image = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
        segmentation_tensor = torch.from_numpy(segmentation_image).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            segmentation_output = torch.sigmoid(segmentation_model(segmentation_tensor))
        mask = (segmentation_output.squeeze().cpu().numpy() >= 0.5).astype(np.uint8)
        mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        area_ratio = float(np.count_nonzero(mask)) / mask.size
        if area_ratio < 0.01:
            severity = "small"
        elif area_ratio < 0.05:
            severity = "medium"
        else:
            severity = "large"

        encoded, mask_png = cv2.imencode(".png", mask * 255)
        if not encoded:
            raise ValueError("Unable to encode mask as PNG.")
        mask_image = base64.b64encode(mask_png.tobytes()).decode("utf-8")
    except Exception as error:
        return {"error": str(error)}

    return {
        "spill_detected": True,
        "confidence": round(confidence, 4),
        "severity": severity,
        "area_ratio": round(area_ratio, 4),
        "mask_image": mask_image,
    }
