import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

FORMAT = "yolov8"
LOCAL_DOWNLOAD_DIR = "./vehicles-dataset"

def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found. Check your .env file.")

    print("Downloading dataset from Roboflow Universe...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-100").project("vehicles-q0x2v")
    dataset = project.version(1).download(model_format=FORMAT, location=LOCAL_DOWNLOAD_DIR)

    print(f"Dataset downloaded locally to: {dataset.location}")

if __name__ == "__main__":
    main()