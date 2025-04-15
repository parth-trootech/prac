import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL")

    # API Base URL
    BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

    # File Storage Paths
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    TEMP_FOLDERS_PATH = os.getenv("TEMP_FOLDERS_PATH", "temp_folders")

    # Logging Level
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # (NEW) Logging level from environment

    # Model Path (New)
    MODEL_PATH = os.getenv("MODEL_PATH", "fine_tuned_resnet_mnist.pth")  # (NEW) Model path from environment

    # Segmentation Configurations
    LINE_THRESHOLD = int(os.getenv("LINE_THRESHOLD", 20))  # (NEW) Configurable line height threshold
    MIN_SEGMENT_HEIGHT = int(os.getenv("MIN_SEGMENT_HEIGHT", 10))  # (NEW) Configurable minimum segment height

    @staticmethod
    def ensure_directories():
        """Ensure required directories exist."""
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        os.makedirs(Config.TEMP_FOLDERS_PATH, exist_ok=True)


# Ensure necessary directories exist
Config.ensure_directories()
