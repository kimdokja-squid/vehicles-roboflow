FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_dataset.py .
COPY train_model.py .
COPY predict.py .
COPY yolov8n.pt .
COPY index.html .

CMD ["python", "train_model.py"]
