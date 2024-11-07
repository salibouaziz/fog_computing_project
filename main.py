from ultralytics import YOLO

def detect_object(image_path, class_id, model_path='yolov8n.pt'):
    
    model = YOLO(model_path)

    # Perform detection on the image
    results = model(image_path)

    # List to store the filtered detections
    object_detections = []

    # Loop through the results and filter by class ID
    for result in results:
        for box in result.boxes:
            if int(box.cls) == class_id:
                object_detections.append(box)

    # Save results with only the filtered detections
    if object_detections:
        result.boxes = object_detections
        result.save()

    return object_detections



# ============= Example usage =============

class_id = 2     # 2 for cars, 0 for persons etc..
image_path = 'car_accident.jpg'

car_detections = detect_object(image_path, class_id)
