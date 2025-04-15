import logging
import os
import uuid

import aiofiles
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.backend.schemas import PredictionRequest, UserCreate, UserLogin
from app.config import Config
from app.db.main import get_db
from app.db.models import User, ImageUpload, PredictionResult
from app.image_processing.predict import predict_all_digits
from app.image_processing.segmentation import segment_with_resnet

# 1. Disable SQLAlchemy engine logs
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# 2. Disable FastAPI / Uvicorn logs
logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)

# 3. (Optional) Disable ALL logging globally
logging.disable(logging.CRITICAL)

# Ensure upload directory exists
UPLOAD_DIR = Config.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize FastAPI
app = FastAPI()


# Utility functions for password hashing
def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


@app.post("/signup")
async def signup(user: UserCreate, db: AsyncSession = Depends(get_db)):
    async with db as session:
        result = await session.execute(select(User).where(User.user_email == user.user_email))
        db_user = result.scalars().first()

        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = hash_password(user.user_password)
        new_user = User(user_email=user.user_email, user_password=hashed_password)
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

        return {"message": "User created successfully"}


@app.post("/login")
async def login(user: UserLogin, db: AsyncSession = Depends(get_db)):
    async with db as session:
        result = await session.execute(select(User).where(User.user_email == user.user_email))
        db_user = result.scalars().first()

        if not db_user or not verify_password(user.user_password, db_user.user_password):
            raise HTTPException(status_code=400, detail="Invalid credentials")

        return {"message": "Login successful", "user_id": db_user.user_id, "user_email": db_user.user_email}


@app.post("/upload_image")
async def upload_image(
        user_id: int = Form(...),
        image: UploadFile = File(...),
        db: AsyncSession = Depends(get_db)
):
    try:
        filename = f"{uuid.uuid4().hex}_{image.filename}"
        file_location = os.path.join(UPLOAD_DIR, filename)

        # Save the uploaded image to disk
        async with aiofiles.open(file_location, "wb") as buffer:
            content = await image.read()
            await buffer.write(content)

        async with db as session:
            image_upload = ImageUpload(user_id=user_id, image_path=file_location)
            session.add(image_upload)
            await session.commit()
            await session.refresh(image_upload)

        return {"message": "Image uploaded successfully", "image_id": image_upload.image_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")


@app.post("/predict")
async def predict(request: PredictionRequest, db: AsyncSession = Depends(get_db)):
    try:
        # Fetch image details from the database
        image_upload_result = await db.execute(
            select(ImageUpload).filter(ImageUpload.image_id == request.image_id)
        )
        image_upload = image_upload_result.scalar_one_or_none()

        if not image_upload:
            raise HTTPException(status_code=404, detail="Image not found")

        image_path = image_upload.image_path
        user_id = image_upload.user_id

        # Create user-specific temp folder
        user_folder = os.path.join(Config.TEMP_FOLDERS_PATH, f"user_{user_id}")
        os.makedirs(user_folder, exist_ok=True)

        # Perform segmentation (Fixed Argument Order)
        output_dir = await segment_with_resnet(image_path, user_id, output_base_dir=user_folder)

        # Predict digits
        output = await predict_all_digits(user_id, Config.TEMP_FOLDERS_PATH)

        # Store prediction in database
        prediction_result = PredictionResult(
            image_id=request.image_id,
            predicted_digit=str(output),
            confidence_score=None
        )
        db.add(prediction_result)
        await db.commit()
        await db.refresh(prediction_result)

        return {"predicted_digit": output, "prediction_id": prediction_result.prediction_id}

    except Exception as e:
        await db.rollback()  # Ensure rollback if anything fails
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        log_level="critical",  # Only show critical logs
        access_log=False  # Disable request/response logs
    )
