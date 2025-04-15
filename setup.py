from setuptools import setup, find_packages

setup(
    name="app",  # Name of your package
    version="0.1",
    packages=find_packages(),  # Automatically finds sub-packages
    install_requires=[
        "fastapi[all]",
        "uvicorn",
        "streamlit",
        "pydantic-settings",
        "sqlmodel",
        "sqlalchemy",
        "numpy",
        "torch",
        "transformers",
        "pandas",
        "Pillow",
        "matplotlib",
        "alembic",
        "psycopg2-binary",
        "bcrypt",
        "pydantic",
        "python-dotenv",
        "asyncpg",
        "passlib",
        "opencv-python",
        "requests",
        "absl-py",
        "torchvision",
        "uuid"
    ],
)
