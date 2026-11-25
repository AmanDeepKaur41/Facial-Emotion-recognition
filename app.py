"""
Flask Backend for the Facial Emotion Recognition System.

Loads the pre-trained Keras CNN model (models/emotion_model.keras) and
OpenCV Haar Cascade face detector once on startup, exposes the web pages
and the /predict JSON inference endpoint.
"""

import base64
import json
import os
from pathlib import Path

_LOCAL_CACHE = Path(__file__).resolve().parent / ".cache"
(_LOCAL_CACHE / "keras").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(_LOCAL_CACHE / "keras"))

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "emotion_model.keras"
LABELS_PATH = MODELS_DIR / "class_labels.json"

DEFAULT_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = 48

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

# Global inference artifacts loaded once at startup
emotion_model = None
model_load_error: str | None = None
class_labels: list[str] = DEFAULT_LABELS.copy()

if hasattr(cv2, "CascadeClassifier"):
    face_cascade_primary = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_cascade_alt = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    )
else:
    # No cascade classifier available
    face_cascade_primary = None
    face_cascade_alt = None


def load_artifacts() -> None:
    """Load the trained Keras model and ordered emotion labels once."""
    global emotion_model, model_load_error, class_labels

    if LABELS_PATH.is_file():
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                loaded_labels = json.load(f)
            if isinstance(loaded_labels, list) and len(loaded_labels) == 7:
                class_labels = [str(lbl) for lbl in loaded_labels]
        except Exception as exc:
            print(f"[WARN] Could not parse {LABELS_PATH}, using defaults: {exc}")

    if not MODEL_PATH.is_file():
        emotion_model = None
        model_load_error = (
            f"Trained model not found at '{MODEL_PATH}'. "
            "Run 'python train_model.py' first to train and save the model."
        )
        print(f"[WARN] {model_load_error}")
        return

    try:
        emotion_model = keras.models.load_model(MODEL_PATH)
        model_load_error = None
        print(f"[INFO] Loaded Keras emotion model from {MODEL_PATH}")
    except Exception as exc:
        emotion_model = None
        model_load_error = f"Failed to load Keras model: {exc}"
        print(f"[ERROR] {model_load_error}")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_uploaded_image(file_storage) -> np.ndarray:
    """Decode uploaded image stream into an OpenCV BGR array with EXIF orientation fix."""
    pil_img = Image.open(file_storage.stream)
    pil_img = ImageOps.exif_transpose(pil_img)
    rgb_img = pil_img.convert("RGB")
    bgr_img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
    return bgr_img


def detect_faces(gray_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect faces using OpenCV Haar Cascades.
    If no cascade classifier is available, return an empty list.
    """
    h, w = gray_img.shape[:2]
    if h == 0 or w == 0:
        return []

    primary_loaded = (
        face_cascade_primary is not None and not face_cascade_primary.empty()
    )
    alt_loaded = face_cascade_alt is not None and not face_cascade_alt.empty()
    if not primary_loaded and not alt_loaded:
        return []

    equalized = cv2.equalizeHist(gray_img)

    # Upscale tiny images (such as direct 48x48 FER-2013 samples) so Haar cascades can detect faces
    min_dim = min(h, w)
    scale = 1.0
    pad = 0
    detect_img = equalized
    if min_dim < 180:
        scale = 240.0 / float(min_dim)
        resized = cv2.resize(
            equalized,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        pad = 40
        detect_img = cv2.copyMakeBorder(
            resized, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE
        )

    faces = ()
    if primary_loaded:
        strict_faces = face_cascade_primary.detectMultiScale(
            detect_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        relaxed_faces = face_cascade_primary.detectMultiScale(
            detect_img, scaleFactor=1.08, minNeighbors=3, minSize=(30, 30)
        )
        faces = relaxed_faces if len(relaxed_faces) > len(strict_faces) else strict_faces
    if len(faces) == 0 and alt_loaded:
        faces = face_cascade_alt.detectMultiScale(
            detect_img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )

    mapped_boxes: list[tuple[int, int, int, int]] = []
    for (fx, fy, fw, fh) in faces:
        ox = int(round((fx - pad) / scale))
        oy = int(round((fy - pad) / scale))
        ow = int(round(fw / scale))
        oh = int(round(fh / scale))

        ox = max(0, ox)
        oy = max(0, oy)
        ow = min(w - ox, ow)
        oh = min(h - oy, oh)
        if ow >= 12 and oh >= 12:
            mapped_boxes.append((ox, oy, ow, oh))

    # Sort left-to-right, top-to-bottom for deterministic face ordering
    mapped_boxes.sort(key=lambda b: (b[0], b[1]))
    return mapped_boxes


def face_detector_is_loaded() -> bool:
    """Return whether at least one Haar cascade initialized successfully."""
    return any(
        cascade is not None and not cascade.empty()
        for cascade in (face_cascade_primary, face_cascade_alt)
    )


def encode_bgr_to_data_uri(bgr_img: np.ndarray, ext: str = ".jpg") -> str:
    success, buffer = cv2.imencode(ext, bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        raise ValueError("Failed to encode image buffer.")
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def preprocess_face_for_model(gray_img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop detected face, resize to 48x48, normalize [0, 1], and reshape to (1, 48, 48, 1)."""
    x, y, w, h = bbox
    face_roi = gray_img[y : y + h, x : x + w]
    interp = cv2.INTER_AREA if min(w, h) > IMG_SIZE else cv2.INTER_CUBIC
    resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE), interpolation=interp)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=(0, -1))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analysis")
def analysis():
    return render_template(
        "analysis.html",
        model_ready=(emotion_model is not None),
    )


@app.route("/health")
def health():
    detector_loaded = face_detector_is_loaded()
    return jsonify(
        {
            "status": "ok" if emotion_model is not None and detector_loaded else "degraded",
            "model_loaded": emotion_model is not None,
            "model_error": model_load_error,
            "face_detector_loaded": detector_loaded,
            "class_labels": class_labels,
        }
    )


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    if emotion_model is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": model_load_error or "Emotion recognition model is not loaded.",
                }
            ),
            503,
        )

    if not face_detector_is_loaded():
        return (
            jsonify(
                {
                    "success": False,
                    "error": "OpenCV Haar Cascade face detector is unavailable.",
                }
            ),
            503,
        )

    if "image" not in request.files:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "No image file provided. Please upload a JPG, JPEG, or PNG image.",
                }
            ),
            400,
        )

    file = request.files["image"]
    if not file or file.filename == "":
        return (
            jsonify(
                {
                    "success": False,
                    "error": "No image selected. Please choose a file to analyze.",
                }
            ),
            400,
        )

    if not allowed_file(file.filename):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Unsupported file format. Please upload a JPG, JPEG, or PNG image.",
                }
            ),
            400,
        )

    try:
        bgr_img = decode_uploaded_image(file)
    except Exception:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Invalid or corrupted image file. Please upload a valid JPG or PNG image.",
                }
            ),
            400,
        )

    try:
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray_img)

        if not faces:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "No face was detected in this image. Please try a clearer, "
                            "front-facing photo with adequate lighting."
                        ),
                    }
                ),
                422,
            )

        annotated_img = bgr_img.copy()
        img_h, img_w = annotated_img.shape[:2]
        # Scale drawing thickness & font relative to image size
        box_thickness = max(2, int(round(min(img_h, img_w) / 200)))
        font_scale = max(0.5, min(1.2, min(img_h, img_w) / 500.0))

        face_results = []
        # Distinct vibrant BGR colors per face bounding box
        palette = [
            (246, 130, 59),   # Blue accent
            (236, 72, 153),   # Pink/Purple accent
            (16, 185, 129),   # Emerald
            (245, 158, 11),   # Amber
        ]

        for idx, bbox in enumerate(faces, start=1):
            x, y, w, h = bbox
            face_tensor = preprocess_face_for_model(gray_img, bbox)
            raw_probs = emotion_model.predict(face_tensor, verbose=0)[0]

            percentages = [round(float(p) * 100.0, 2) for p in raw_probs]
            dominant_idx = int(np.argmax(raw_probs))
            dominant_emotion = class_labels[dominant_idx]
            confidence = round(float(raw_probs[dominant_idx]) * 100.0, 2)

            probabilities = {
                label: pct for label, pct in zip(class_labels, percentages)
            }

            # Draw bounding box and label badge on annotated image
            color = palette[(idx - 1) % len(palette)]
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, box_thickness)

            tag = f"#{idx} {dominant_emotion} ({confidence:.1f}%)"
            (tw, th), baseline = cv2.getTextSize(
                tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, box_thickness - 1)
            )
            tag_y_top = max(0, y - th - baseline - 8)
            tag_y_bottom = max(th + baseline + 8, y)
            cv2.rectangle(
                annotated_img,
                (x, tag_y_top),
                (min(img_w, x + tw + 12), tag_y_bottom),
                color,
                -1,
            )
            cv2.putText(
                annotated_img,
                tag,
                (x + 6, tag_y_bottom - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                max(1, box_thickness - 1),
                cv2.LINE_AA,
            )

            # Encode individual cropped face thumbnail
            face_crop_bgr = bgr_img[y : y + h, x : x + w]
            face_crop_preview = cv2.resize(
                face_crop_bgr, (140, 140), interpolation=cv2.INTER_CUBIC
            )
            face_crop_uri = encode_bgr_to_data_uri(face_crop_preview, ext=".jpg")

            face_results.append(
                {
                    "face_id": idx,
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "dominant_emotion": dominant_emotion,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "face_crop": face_crop_uri,
                }
            )

        annotated_uri = encode_bgr_to_data_uri(annotated_img, ext=".jpg")

        return jsonify(
            {
                "success": True,
                "face_count": len(face_results),
                "image_dimensions": {"width": img_w, "height": img_h},
                "class_labels": class_labels,
                "annotated_image": annotated_uri,
                "faces": face_results,
            }
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Server error during emotion analysis: {exc}",
                }
            ),
            500,
        )


@app.errorhandler(413)
def file_too_large(_error):
    return (
        jsonify(
            {
                "success": False,
                "error": "Uploaded image exceeds the 10 MB size limit.",
            }
        ),
        413,
    )


load_artifacts()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
