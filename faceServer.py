from flask import Flask, request, jsonify
import os
import uuid
import base64
import tempfile
import requests
from pathlib import Path
from training.train_script import train_model_from_directory  # Import your refactored function
from face_recognition import preprocessing
from PIL import Image
from io import BytesIO
import joblib
 
app = Flask(__name__)

MODEL_BASE_PATH = "model/face_rec_6"





@app.route("/train", methods=["POST"])
def train():
    try:
        data = request.get_json()
        model_name = data.get("modelName")
        students_data = data.get("data", [])

        if not model_name or not students_data:
            return jsonify({"error": "Missing modelName or data"}), 400

        temp_dataset_dir = tempfile.mkdtemp()

        for item in students_data:
            student_id = item["studentId"]
            image_url = item["imageUrl"]

            student_dir = Path(temp_dataset_dir) / student_id
            student_dir.mkdir(parents=True, exist_ok=True)

            response = requests.get(image_url)
            if response.status_code == 200:
                file_path = student_dir / f"{uuid.uuid4().hex}.jpg"
                with open(file_path, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download image from {image_url}")

        model_output_path = os.path.join(MODEL_BASE_PATH, model_name, "face_recogniser.pkl")
        trained_model_path = train_model_from_directory(temp_dataset_dir, model_output_path)

        return jsonify({
            "success": True,
            "modelPath": trained_model_path,
            "modelName": model_name,
        })

    except Exception as e:
        print("Error during training:", str(e))
        return jsonify({"error": str(e)}), 500






preprocess = preprocessing.ExifOrientationNormalize()
@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.get_json()
        model_path = data.get("modelPath")  # This is now full path like "model/face_rec_6/CSE-A/face_recogniser.pkl"
        base64_image = data.get("base64Image")

        if not model_path or not base64_image:
            return jsonify({"error": "modelPath and image are required"}), 400

        # Ensure path is normalized (cross-platform safe)
        model_path = os.path.normpath(model_path)

        if not os.path.exists(model_path):
            return jsonify({"error": f"Model not found at {model_path}"}), 404

        # Load model
        model = joblib.load(model_path)

        # Decode and preprocess the image
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_bytes))
        image = preprocess(image).convert('RGB')

        # Inference
        results = model(image)
        verified = []

        for face in results:
            label = face.top_prediction.label
            if label != "Unknown":
                verified.append(label)

        return jsonify({"verifiedStudents": verified})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
