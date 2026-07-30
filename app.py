import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

MODEL_PATH = "./runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    results = model.predict(source=image, verbose=False)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class": class_name,
            "confidence": conf
        })

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)