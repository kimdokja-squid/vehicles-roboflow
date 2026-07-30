FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_dataset.py .
COPY train_model.py .
COPY predict.py .
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]