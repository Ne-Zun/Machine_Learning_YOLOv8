from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# train the model with CPU
results = model.train(data="mydata.yaml", epochs=100,
    imgsz=640,
    batch=8,
    augment=True )

