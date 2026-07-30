from ultralytics import YOLO

DATA_YAML = "./vehicles-dataset/data.yaml"
MODEL_CHECKPOINT = "yolov8n.pt"
EPOCHS = 50
IMG_SIZE = 320
BATCH_SIZE = 4

def main():
    print(f"Starting training using {DATA_YAML} ...")
    model = YOLO(MODEL_CHECKPOINT)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE
    )
    print("Training complete. Check runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()