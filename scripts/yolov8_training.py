from ultralytics import YOLO
import torch
import os
from config import YOLO_MODEL_NAME
import json 

DATA_PATH = "./dataset/VehicleCount/data.yaml"

def train_yolov8():
    # Load the model
    model = YOLO(YOLO_MODEL_NAME)
    print(f"Model loaded: {YOLO_MODEL_NAME}")
    # Train
    results = model.train(
        data=DATA_PATH,
        epochs=50,
        imgsz=640,
        device=0,
        batch=16,
        workers=4,
        single_cls=True  # Optional: Forces the model to treat data as single-class
    )
    print("Training complete.")

def evaluate_yolov8(model_path):
    # model = YOLO(YOLO_MODEL_NAME)
    model = YOLO(model_path)

    # Run validation on the 'test' split
    metrics = model.val(
        data=DATA_PATH,
        split='test'  # <--- Critical: tells it to use the test set
    )

    print(f"Test mAP50: {metrics.box.map50}")
    print(f"Test mAP50-95: {metrics.box.map}")

def predict(model_path, test_images_dir):
    # 1. Load the model
    # model = YOLO(YOLO_MODEL_NAME)
    model = YOLO(model_path)

    # 2. Run prediction on the whole folder
    SAVE_DIR = f"./results/test"
    os.makedirs(SAVE_DIR, exist_ok=True)
    results = model.predict(source=test_images_dir, save=True, conf=0.5, iou=0.65, project=os.path.dirname(SAVE_DIR), name=os.path.basename(SAVE_DIR))

    print(f"Check the folder: {results[0].save_dir}")

def count_objects(model_path = None, image_path = None, model = None):
    # Load the model
    # model = YOLO(YOLO_MODEL_NAME)
    if not model:
        model = YOLO(model_path)

    # Run inference
    results = model(image_path, conf=0.3, iou=0.65, verbose=False)
    
    # Get the first result (since we only passed one image)
    detections = results[0]

    # CHECK: Is this an OBB model or a Normal Box model?
    if detections.obb is not None:
        # It's an OBB model (Rotated Boxes)
        object_count = len(detections.obb)
    else:
        # It's a Standard model (Rectangles)
        object_count = len(detections.boxes)
    
    return object_count


def create_image_json(model, images_dir, output_path, verbose=False):
    d = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
    }
    for i, image_path in enumerate(os.listdir(images_dir)):
        full_image_path = os.path.join(images_dir, image_path)
        count = count_objects(model=model, image_path=full_image_path)
        if count < 10: d["B"].append(full_image_path)
        elif count < 30: d["C"].append(full_image_path)
        elif count < 50: d["A"].append(full_image_path)
        else: d["D"].append(full_image_path)
        if verbose:
            if i % 100 == 0:
                print(f"Processed {i} files")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4)

if __name__ == "__main__":
    print(f'GPU Available: {torch.cuda.is_available()}')
    print(torch.__version__)
    # train_yolov8()
    # evaluate_yolov8("runs/detect/train-2/weights/best.pt")
    # predict("runs/detect/train-2/weights/best.pt", "test_img.jpg")
    # count = count_objects("runs/detect/train-2/weights/best.pt", "test_img.jpg")
    # print(count)
    # img_num = 24
    # for i in os.listdir("./frames"):
    #     img_num = i[-2:]
    #     predict(f"data/models/IMG_00{img_num}/img_00{img_num}.pt", f"data/models/IMG_00{img_num}/frames", img_num)
    # image_dir = "./frames/IMG_0028"
    # for img_file in os.listdir(image_dir):
    #     if img_file.endswith(('.png', '.jpg', '.jpeg')):
    #         image_path = os.path.join(image_dir, img_file)
    #         count = count_objects(image_path, "runs/obb/train10/weights/best.pt")
    #         print(f"Image: {img_file}, Object Count: {count}")
    model = YOLO("runs/detect/train-2/weights/best.pt")
    create_image_json(model, "dataset/VehicleCount/train/images", "saved_image_path.json", verbose=True)
    