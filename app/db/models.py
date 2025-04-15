from datetime import datetime
from typing import List, Optional

from sqlalchemy import create_engine
from sqlmodel import Field, SQLModel, Relationship

from app.config import Config


# Users Table
class User(SQLModel, table=True):
    __tablename__ = "users"

    user_id: int = Field(default=None, primary_key=True)
    user_email: str = Field(index=True, unique=True, nullable=False)
    user_password: str = Field(nullable=False)

    # Relationship
    image_uploads: List["ImageUpload"] = Relationship(back_populates="user")


# Image_Uploads Table
class ImageUpload(SQLModel, table=True):
    __tablename__ = "image_uploads"

    image_id: int = Field(default=None, primary_key=True, index=True)
    user_id: int = Field(foreign_key="users.user_id")
    image_path: str = Field(nullable=False, index=True)
    # noinspection PyDeprecation
    upload_time: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    # Relationship
    user: User = Relationship(back_populates="image_uploads")
    predictions: List["PredictionResult"] = Relationship(back_populates="image_upload")


# noinspection PyDeprecation
class PredictionResult(SQLModel, table=True):
    __tablename__ = "prediction_results"

    prediction_id: int = Field(default=None, primary_key=True, index=True)
    image_id: int = Field(foreign_key="image_uploads.image_id")
    predicted_digit: str = Field(nullable=False)
    confidence_score: Optional[float] = Field(default=None, nullable=True)
    prediction_time: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    image_upload: Optional[ImageUpload] = Relationship(back_populates="predictions")


# Use a synchronous engine for table creation
sync_engine = create_engine(Config.DATABASE_URL.replace("postgresql+asyncpg", "postgresql"))


# Create tables if they do not exist
def init_db():
    """Initialize the database."""
    SQLModel.metadata.create_all(sync_engine)


# Call the function to ensure tables are created at startup
init_db()
