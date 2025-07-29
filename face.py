from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.formparsers import MultiPartParser
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

import joblib
from face_recognition import preprocessing
from PIL import Image
import base64
from io import BytesIO
from torchvision import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import shutil

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
# Import your module-level helpers as needed
from training.train import (
    dataset_to_embeddings,
    compute_class_centers,
    FaceFeaturesExtractor,
    FaceRecogniser,
    train,
    MODEL_DIR_PATH
)

app = FastAPI()
class VerifyRequest(BaseModel):
    base64Image: str
    modelPath: str = "model/face_recogniser.pkl"

class TrainRequest(BaseModel):
    grid_search: int = 0
    dataset_path: str = "datasets/CSE-A"

@app.post("/verify")
async def verify(data: VerifyRequest):
    try:
        preprocess = preprocessing.ExifOrientationNormalize()
        base64_image = data.base64Image
        model_path = data.modelPath
        print(model_path)
        if not model_path or not base64_image:
            raise HTTPException(status_code=400, detail="modelPath and image are required")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")

        # Load model
        model = joblib.load(model_path)

        # Decode and preprocess image
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_bytes))
        image = preprocess(image).convert("RGB")
        image.save("images/image.jpg")

        # Inference
        results = model(image)
        verified = [face.top_prediction.label for face in results]

        return {"verifiedStudents": verified}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_controller(data: TrainRequest):
    try:
        grid_search = bool(data.grid_search)
        dataset_path = data.dataset_path or "datasets/CSE-A"

        if not dataset_path or not os.path.isdir(dataset_path):
            raise HTTPException(status_code=400, detail="Valid dataset_path required")

        features_extractor = FaceFeaturesExtractor()
        dataset = datasets.ImageFolder(dataset_path)
        embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
        class_to_idx = dataset.class_to_idx

        embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )

        class Args: pass
        args = Args()
        args.grid_search = grid_search
        clf = train(args, embeddings_train, labels_train)

        class_centers = compute_class_centers(embeddings_train, labels_train, class_to_idx)
        labels_pred = clf.predict(embeddings_test)
        report = metrics.classification_report(labels_test, labels_pred, output_dict=True)

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        if not os.path.isdir(MODEL_DIR_PATH):
            os.mkdir(MODEL_DIR_PATH)
        model_path = os.path.join(MODEL_DIR_PATH, 'face_recogniser.pkl')
        joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class, class_centers), model_path)

        return {
            "message": f"Model trained and saved to {model_path}",
            "classification_report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
BASE_MODEL_DIR = "final_model"






@app.post("/upload-model")
async def upload_model(
    model: UploadFile = File(...),
    department: str = Form(...), 
    division: str = Form(...)
):
    # Validate file extension
    if not model.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Only .pkl files are allowed.")

    # Construct directory path
    model_dir_name = f"{department}-{division}"
    model_dir_path = os.path.join(BASE_MODEL_DIR, model_dir_name)

    # Create directory if it doesn't exist
    os.makedirs(model_dir_path, exist_ok=True)

    # Final path for saved model file
    model_file_path = os.path.join(model_dir_path, "model.pkl")

    try:
        # Stream the uploaded file to disk efficiently
        with open(model_file_path, "wb") as f:
            # Read in chunks to avoid large memory use
            while True:
                chunk = await model.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    return JSONResponse(content={"model_path": model_file_path}, status_code=200)


