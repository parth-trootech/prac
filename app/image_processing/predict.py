import asyncio  # Async processing
import os
import re

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
from torchvision.transforms.functional import invert

from app.config import Config


async def load_model():
    """Load the ResNet model asynchronously."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)

    model_path = Config.MODEL_PATH
    if not os.path.exists(model_path):
        return None

    model_state = await asyncio.to_thread(torch.load, model_path, map_location=device)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


async def transform_image(image_path):
    """Apply transformations asynchronously."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Lambda(lambda x: invert(x)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = await asyncio.to_thread(Image.open, image_path)
    image = image.convert("L")

    return transform(image).unsqueeze(0)


def extract_number(filename):
    """Extract numeric parts from filenames for sorting."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0  # Default to 0 instead of float('inf')


async def predict_digits(model, image_folder, device):
    """Predict digits from images asynchronously."""
    if not os.path.exists(image_folder):
        return ""

    predictions = []
    files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=extract_number
    )

    for filename in files:
        image_path = os.path.join(image_folder, filename)
        image = await transform_image(image_path)
        image = image.to(device)

        with torch.no_grad():
            output = model(image)
            predicted_digit = torch.argmax(output, dim=1).item()
            predictions.append(str(predicted_digit))

    return "".join(predictions)


async def predict_all_digits(user_id, base_folder):
    """Main function to predict digits for a user asynchronously."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = await load_model()
    if model is None:
        return ""

    user_folder = os.path.join(base_folder, f"user_{user_id}")
    if not os.path.exists(user_folder):
        return ""

    line_folders = sorted(
        [folder for folder in os.listdir(user_folder) if folder.startswith("temp_folder_")],
        key=extract_number
    )

    tasks = [
        predict_digits(model, os.path.join(user_folder, folder), device) for folder in line_folders
    ]

    results = await asyncio.gather(*tasks)
    return "_".join(results)
