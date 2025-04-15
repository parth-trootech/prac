import asyncio  # Async processing
import os

import cv2
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

from ..config import Config


async def create_segmentation_model():
    """Load a pretrained ResNet-18 model asynchronously and modify it for segmentation."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    return model


async def preprocess_image(image_path):
    """Preprocess input image asynchronously for ResNet-18."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = await asyncio.to_thread(Image.open, image_path)

    return transform(image).unsqueeze(0)


async def segment_with_resnet(image_path, user_id, output_base_dir=None):
    """Segment handwritten text into lines using a modified ResNet-18 asynchronously."""
    output_base_dir = output_base_dir or os.path.join(Config.TEMP_FOLDERS_PATH, f"user_{user_id}")
    os.makedirs(output_base_dir, exist_ok=True)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = sorted([cv2.boundingRect(contour) for contour in contours], key=lambda x: x[1])
    line_threshold = Config.LINE_THRESHOLD

    lines = []
    current_line = []
    prev_y = -line_threshold

    for box in bounding_boxes:
        x, y, w, h = box
        if abs(y - prev_y) > line_threshold:
            if current_line:
                lines.append(sorted(current_line, key=lambda x: x[0]))
            current_line = [box]
        else:
            current_line.append(box)
        prev_y = y

    if current_line:
        lines.append(sorted(current_line, key=lambda x: x[0]))

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Ensure correct color format

    async def save_segment(segment, segment_path):
        """Asynchronously save image segment."""
        await asyncio.to_thread(cv2.imwrite, segment_path, segment)

    async def process_line(line_idx, line):
        """Process each segmented line asynchronously."""
        line_folder = os.path.join(output_base_dir, f"temp_folder_{line_idx}")
        os.makedirs(line_folder, exist_ok=True)

        tasks = [
            save_segment(original_image[y:y + h, x:x + w], os.path.join(line_folder, f"digit_{digit_idx + 1}.png"))
            for digit_idx, (x, y, w, h) in enumerate(line) if h > Config.MIN_SEGMENT_HEIGHT
        ]

        await asyncio.gather(*tasks)

    await asyncio.gather(*(process_line(line_idx, line) for line_idx, line in enumerate(lines)))

    return output_base_dir  # Returns user-specific folder path
