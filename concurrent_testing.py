import asyncio
import time
from pathlib import Path

import httpx

from app.frontend.app import clear_temp_folders

# API ENDPOINTS
LOGIN_URL = "http://localhost:8000/login"
UPLOAD_URL = "http://localhost:8000/upload_image"
PREDICTION_RESULT_URL = "http://localhost:8000/predict"

IMAGE_DIR = Path("test_images")
IMAGE_FILES = sorted(IMAGE_DIR.glob("*.png"))  # Adjust extension if needed

USERS = [
    {"user_email": "test_user_1@hdrs.com", "user_password": "test_user_1"},
    {"user_email": "test_user_2@hdrs.com", "user_password": "test_user_2"},
    {"user_email": "test_user_3@hdrs.com", "user_password": "test_user_3"},
    {"user_email": "test_user_4@hdrs.com", "user_password": "test_user_4"},
    {"user_email": "test_user_5@hdrs.com", "user_password": "test_user_5"},
    {"user_email": "test_user_6@hdrs.com", "user_password": "test_user_6"},
    {"user_email": "test_user_7@hdrs.com", "user_password": "test_user_7"},
    {"user_email": "test_user_8@hdrs.com", "user_password": "test_user_8"},
    {"user_email": "test_user_9@hdrs.com", "user_password": "test_user_9"},
    {"user_email": "test_user_10@hdrs.com", "user_password": "test_user_10"},
    {"user_email": "test_user_11@hdrs.com", "user_password": "test_user_11"},
    {"user_email": "test_user_12@hdrs.com", "user_password": "test_user_12"},
    {"user_email": "test_user_13@hdrs.com", "user_password": "test_user_13"},
    {"user_email": "test_user_14@hdrs.com", "user_password": "test_user_14"},
    {"user_email": "test_user_15@hdrs.com", "user_password": "test_user_15"},
]


async def login_user(client, user):
    """Attempt to log in the user and return (success, user_id)."""
    try:
        response = await client.post(LOGIN_URL, json=user, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        user_id = data.get("user_id")

        if user_id:
            print(f"Logged in: {user['user_email']} (User ID: {user_id})")
            return True, user_id
        else:
            print(f"❌ Login response missing user_id: {response.text}")
            return False, None
    except httpx.HTTPStatusError as e:
        print(f"Login failed: {user['user_email']} - {e.response.text}")
        return False, None
    except httpx.ReadTimeout:
        print(f"Login timed out for {user['user_email']}")
        return False, None
    except Exception as e:
        print(f"Unexpected error in login: {str(e)}")
        return False, None


async def upload_image(client, image_path, user_id):
    """Upload an image and return its server-side path."""
    try:
        with open(image_path, "rb") as image_file:
            files = {"image": (image_path.name, image_file, "image/png")}
            data = {"user_id": str(user_id)}
            response = await client.post(UPLOAD_URL, files=files, data=data, timeout=30.0)
            response.raise_for_status()

            data = response.json()
            image_id = data.get("image_id")
            print(f"Uploaded: {image_path.name} (Image ID: {image_id})")
            return image_id
    except httpx.HTTPStatusError as e:
        print(f"Upload failed: {image_path.name} - {e.response.text}")
    except httpx.ReadTimeout:
        print(f"Upload timed out for {image_path.name}")
    except Exception as e:
        print(f"Unexpected error in upload: {str(e)}")
    return None


async def get_prediction(client, image_id):
    """Fetch prediction results for the uploaded image."""
    try:
        response = await client.post(
            PREDICTION_RESULT_URL,
            json={"image_id": image_id},
            timeout=60.0  # Increased timeout for model processing
        )
        response.raise_for_status()

        data = response.json()
        print(f"Prediction Result: {data}")
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
    except httpx.ReadTimeout:
        print(f"Prediction timed out for image_id: {image_id}")
    except Exception as e:
        print(f"Unexpected error in prediction: {str(e)}")


async def process_user(client, user, image_path):
    """Login, upload image, and get predictions."""
    try:
        start = time.perf_counter()
        logged_in, user_id = await login_user(client, user)
        print(f"Login Time: {time.perf_counter() - start:.2f} sec")

        if not logged_in or user_id is None:
            return

        start = time.perf_counter()
        image_id = await upload_image(client, image_path, user_id)
        print(f"Upload Time: {time.perf_counter() - start:.2f} sec")

        if not image_id:
            return

        start = time.perf_counter()
        await get_prediction(client, image_id)
        print(f"Prediction Time: {time.perf_counter() - start:.2f} sec")
    except Exception as e:
        print(f"Unexpected error in process_user: {str(e)}")


async def main():
    """Run tests for all users and images concurrently, measure execution time."""
    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [process_user(client, USERS[i % len(USERS)], img) for i, img in enumerate(IMAGE_FILES)]
        await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())

clear_temp_folders()
