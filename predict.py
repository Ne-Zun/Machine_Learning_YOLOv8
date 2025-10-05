from ultralytics import YOLO
from PIL import Image

# Load a model
model = YOLO("F:/MACHINE_TRAFFIC/runs/detect/train3/weights/best.pt")

results = model("F:/MACHINE_TRAFFIC/ztest1.jpg")


for r in results:
    print(r.boxes)
    im_array = r.plot()
    im=Image.fromarray(im_array[...,  ::-1])
    im.show()
    im.save("kq.jpg")