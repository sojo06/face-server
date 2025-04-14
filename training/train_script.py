# training/train_script.py

import os
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
from face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser

def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])
    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(f"Processing: {img_path}")
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print(f"Could not find face in {img_path}")
            continue
        if embedding.shape[0] > 1:
            print(f"Multiple faces detected in {img_path}, selecting the most confident.")
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)
    return np.stack(embeddings), labels

def compute_class_centers(embeddings, labels, class_to_idx):
    class_centers = {}
    for class_name, class_idx in class_to_idx.items():
        class_embeddings = np.array([embeddings[i] for i in range(len(labels)) if labels[i] == class_idx])
        class_centers[class_idx] = np.mean(class_embeddings, axis=0)
    return class_centers

def train_model_from_directory(dataset_path, model_output_path):
    features_extractor = FaceFeaturesExtractor()

    dataset = datasets.ImageFolder(dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    class_to_idx = dataset.class_to_idx

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000)
    clf.fit(X_train, y_train)

    class_centers = compute_class_centers(X_train, y_train, class_to_idx)
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    recognizer = FaceRecogniser(features_extractor, clf, idx_to_class, class_centers)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(recognizer, model_output_path)
    print(f"Saved model to {model_output_path}")
    return model_output_path
