# Handwritten Digit Recognition System

## 🚀 Overview

This project is a **Handwritten Digit Recognition System** built using **FastAPI** (backend) and **Streamlit** (
frontend). The model is based on **ResNet-18**, fine-tuned on the MNIST dataset. Users can upload handwritten digit
images, and the system will predict the digits using deep learning techniques.

## 🏗 Project Structure

```
project/
│── app/
│   ├── backend/
│   │   ├── app.py           # FastAPI server
│   │   ├── schemas.py       # To validate request bodies
│   ├── frontend/
│   │   ├── app.py           # Streamlit frontend for user interaction
│   ├── config.py            # Configuration settings
│── db/
│   ├── main.py              # Database Configuration
│   ├── models.py            # Models for database
│── image_processing/
│   ├── predict.py           # Image prediction
│   ├── segmentation.py      # Image segmentation logic
│── .env                     # Environment variables
│── Dockerfile               # Docker setup
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set up the Database (PostgreSQL)

```bash
# Start PostgreSQL and create a database
psql -U postgres -c "CREATE DATABASE digit_recognition;"
```

Make sure your `.env` file contains the correct database URL:

```
DATABASE_URL=postgresql://username:password@localhost:5432/digit_recognition
```

### 4️⃣ Run Backend (FastAPI)

```bash
uvicorn app.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5️⃣ Run Frontend (Streamlit)

```bash
streamlit run app/frontend/app.py
```

## 🚀 API Endpoints

| Method | Endpoint    | Description                 |
|--------|-------------|-----------------------------|
| `POST` | `/signup/`  | Register a new user         |
| `POST` | `/login/`   | Login a registered user     |
| `POST` | `/upload/`  | Upload a handwritten image  |
| `GET`  | `/predict/` | Predict digit from an image |

---

## 📜 License

This project is licensed under the MIT License.

---

## ✨ Contributors

- Parth-Dobariya

---

## 🛠 Future Enhancements

✅ **Improve concurrency handling**
✅ **Deploy using Docker

---

Enjoy coding! 🚀

