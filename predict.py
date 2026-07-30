import sys
from ultralytics import YOLO

MODEL_PATH = "./runs/detect/train/weights/best.pt"

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        return

    image_path = sys.argv[1]
    model = YOLO(MODEL_PATH)
    model.predict(source=image_path, save=True)
    print("Prediction complete. Check runs/detect/predict/ for output image.")

if __name__ == "__main__":
    main()