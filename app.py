# import logging
#
# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from firebase_admin import credentials, firestore, initialize_app
# from datetime import datetime
# import time
# import base64
#
# # Flask app initialization
# app = Flask(__name__)
#
# # Firebase setup
# cred = credentials.Certificate(
#     'D:/FYP project/cane2scan_python/database/project-fyp-7e8c1-firebase-adminsdk-ehwmg-d03dca4472.json')
# initialize_app(cred)
# db = firestore.client()
#
# # YOLO models
# models = {
#     "yolov8s": YOLO('D:/FYP project/cane2scan_python/trained/yolov8s/weights/best.pt'),
#     "yolov8n": YOLO('D:/FYP project/cane2scan_python/trained/yolov8n/weights/best.pt'),
#     "yolov8m": YOLO('D:/FYP project/cane2scan_python/trained/yolov8m/weights/best.pt'),
#     "yolov8l": YOLO('D:/FYP project/cane2scan_python/trained/yolov8l/weights/best.pt'),
# }
#
# class_labels = {0: 'Healthy Leaf', 1: 'WLD'}
#
# @app.route('/test', methods=['GET'])
# def test_connection():
#     return jsonify({"status": "success", "message": "Connection successful!"}), 200
#
#
# def is_duplicate(box, existing_boxes, threshold=0.5):
#     """Check if two bounding boxes are duplicates using IoU."""
#     x1, y1, x2, y2 = box
#     for ex_x1, ex_y1, ex_x2, ex_y2 in existing_boxes:
#         inter_x1 = max(x1, ex_x1)
#         inter_y1 = max(y1, ex_y1)
#         inter_x2 = min(x2, ex_x2)
#         inter_y2 = min(y2, ex_y2)
#         inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#         box_area = (x2 - x1) * (y2 - y1)
#         ex_box_area = (ex_x2 - ex_x1) * (ex_y2 - ex_y1)
#         union_area = box_area + ex_box_area - inter_area
#         iou = inter_area / union_area if union_area > 0 else 0
#         if iou > threshold:
#             return True
#     return False
#
#
# def crop_image_with_padding(image, box, padding=20):
#     """Crop an image with padding, ensuring it stays within bounds."""
#     x1, y1, x2, y2 = box
#     x1 = max(0, x1 - padding)
#     y1 = max(0, y1 - padding)
#     x2 = min(image.shape[1], x2 + padding)
#     y2 = min(image.shape[0], y2 + padding)
#     return image[y1:y2, x1:x2]
#
#
# def create_collage(original, annotated):
#     """Combine original and annotated images into a collage."""
#     # Resize both images to the same height
#     height = max(original.shape[0], annotated.shape[0])
#     width = original.shape[1] + annotated.shape[1]
#     collage = np.zeros((height, width, 3), dtype=np.uint8)
#
#     # Place original and annotated side by side
#     collage[:original.shape[0], :original.shape[1]] = original
#     collage[:annotated.shape[0], original.shape[1]:] = annotated
#     return collage
#
#
# @app.route('/detect', methods=['POST'])
# def detect():
#     """Detect WLD in an uploaded image."""
#     try:
#         # Get the uploaded image
#         image_file = request.files.get('file')
#         if not image_file:
#             return jsonify({"error": "No image uploaded"}), 400
#
#         # Read the image
#         image_bytes = image_file.read()
#         original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
#         if original_image is None:
#             return jsonify({"error": "Invalid image format"}), 400
#
#         # Start timer for processing
#         start_time = time.time()
#
#         image_copy = original_image.copy()
#         aggregated_detections = []
#         existing_boxes = []
#         cropped_images_base64 = []
#
#         # Dictionary to store model-specific detection counts and confidence
#         model_detections = {model_name: {"count": 0, "confidence": 0.0} for model_name in models.keys()}
#
#         # Process all models
#         results_by_model = {model_name: model(image_copy) for model_name, model in models.items()}
#
#         # Process detection results for each model
#         for model_name, results in results_by_model.items():
#             model_confidences = []
#             for result in results:
#                 for box in result.boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     conf = float(box.conf[0])
#                     class_id = int(box.cls[0])
#
#                     # Avoid duplicates and only include WLD detections
#                     if conf > 0.5 and class_id == 1 and not is_duplicate((x1, y1, x2, y2), existing_boxes):
#                         aggregated_detections.append((x1, y1, x2, y2, conf, class_id, model_name))
#                         existing_boxes.append((x1, y1, x2, y2))
#                         model_detections[model_name]["count"] += 1
#                         model_confidences.append(conf)
#
#                         # Crop and encode the detected area
#                         cropped_image = crop_image_with_padding(original_image, (x1, y1, x2, y2))
#                         _, buffer = cv2.imencode('.jpg', cropped_image)
#                         cropped_images_base64.append(base64.b64encode(buffer).decode('utf-8'))
#
#             # Calculate average confidence for this model
#             if model_confidences:
#                 model_detections[model_name]["confidence"] = sum(model_confidences) / len(model_confidences)
#
#         # Detection metadata
#         wld_detections = len(aggregated_detections)
#         detection_status = "No WLD Found" if wld_detections == 0 else "WLD Detected"
#         percentage_confidence = sum(d[4] for d in aggregated_detections) / len(
#             aggregated_detections) * 100 if aggregated_detections else 0
#
#         # End timer and calculate processing time
#         end_time = time.time()
#         processing_time = end_time - start_time
#
#         # Draw bounding boxes on a copy of the image
#         for x1, y1, x2, y2, conf, class_id, model_name in aggregated_detections:
#             color = (0, 0, 255)
#             cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
#
#         # Create a collage image
#         collage_image = create_collage(original_image, image_copy)
#
#         # Save the original and collage images to files
#         original_image_path = 'path_to_save_original_image.jpg'
#         collage_image_path = 'path_to_save_collage_image.jpg'
#
#         cv2.imwrite(original_image_path, original_image)
#         cv2.imwrite(collage_image_path, collage_image)
#
#         # Encode the original and collage images to base64 by reading from saved files
#         with open(original_image_path, "rb") as image_file:
#             original_base64 = base64.b64encode(image_file.read()).decode("utf-8")
#
#         with open(collage_image_path, "rb") as image_file:
#             collage_base64 = base64.b64encode(image_file.read()).decode("utf-8")
#
#         # Prepare data to save in Firestore
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         firestore_data = {
#             "status": detection_status,
#             "confidence": f"{percentage_confidence:.2f}%",
#             "wld_count": wld_detections,
#             "processing_time": f"{processing_time:.2f} seconds",
#             "date_time": timestamp,
#             "model_detections": model_detections,
#             "original_image": original_base64,
#             "collage_image": collage_base64,
#             "cropped_images": cropped_images_base64,
#         }
#
#         # Save data in Firestore
#         db.collection("detections").add(firestore_data)
#
#         # Return data to Flutter
#         jsonify ({
#             "message": "Detection completed",
#             "status": detection_status,
#             "confidence": f"{percentage_confidence:.2f}%",
#             "wld_count": wld_detections,
#             "processing_time": f"{processing_time:.2f} seconds",
#             "model_detections": model_detections,
#             "collage_image": collage_base64,
#         })
#
#         # Return the JSON response
#         return jsonify({"message": "Detection completed"}), 200
#     except Exception as e:
#         logging.error(f"Error during detection: {str(e)}")
#         return jsonify({"error": "An error occurred", "details": str(e)}), 500
#
#
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)



import logging

from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time
import base64

# Flask app initialization
app = Flask(__name__)

# YOLO models
models = {
    "yolov8s": YOLO('D:/FYP project/cane2scan_python/trained/yolov8s/weights/best.pt'),
    "yolov8n": YOLO('D:/FYP project/cane2scan_python/trained/yolov8n/weights/best.pt'),
    "yolov8m": YOLO('D:/FYP project/cane2scan_python/trained/yolov8m/weights/best.pt'),
    "yolov8l": YOLO('D:/FYP project/cane2scan_python/trained/yolov8l/weights/best.pt'),
}

class_labels = {0: 'Healthy Leaf', 1: 'WLD'}

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({"status": "success", "message": "Connection successful!"}), 200

def is_duplicate_strict(box, existing_boxes, threshold=0.1):
    """Check if two bounding boxes are duplicates using IoU with a stricter threshold."""
    x1, y1, x2, y2 = box
    for ex_x1, ex_y1, ex_x2, ex_y2 in existing_boxes:
        inter_x1 = max(x1, ex_x1)
        inter_y1 = max(y1, ex_y1)
        inter_x2 = min(x2, ex_x2)
        inter_y2 = min(y2, ex_y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box_area = (x2 - x1) * (y2 - y1)
        ex_box_area = (ex_x2 - ex_x1) * (ex_y2 - ex_y1)
        union_area = box_area + ex_box_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        if iou > threshold:
            return True
    return False

# def is_duplicate(box, existing_boxes, threshold=0.5):
#     """Check if two bounding boxes are duplicates using IoU."""
#     x1, y1, x2, y2 = box
#     for ex_x1, ex_y1, ex_x2, ex_y2 in existing_boxes:
#         inter_x1 = max(x1, ex_x1)
#         inter_y1 = max(y1, ex_y1)
#         inter_x2 = min(x2, ex_x2)
#         inter_y2 = min(y2, ex_y2)
#         inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#         box_area = (x2 - x1) * (y2 - y1)
#         ex_box_area = (ex_x2 - ex_x1) * (ex_y2 - ex_y1)
#         union_area = box_area + ex_box_area - inter_area
#         iou = inter_area / union_area if union_area > 0 else 0
#         if iou > threshold:
#             return True
#     return False


def crop_image_with_padding(image, box, padding=1):
    """Crop an image with padding, ensuring it stays within bounds."""
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    return image[y1:y2, x1:x2]


def create_collage(original, annotated):
    """Combine original and annotated images into a collage."""
    # Resize both images to the same height
    height = max(original.shape[0], annotated.shape[0])
    width = original.shape[1] + annotated.shape[1]
    collage = np.zeros((height, width, 3), dtype=np.uint8)

    # Place original and annotated side by side
    collage[:original.shape[0], :original.shape[1]] = original
    collage[:annotated.shape[0], original.shape[1]:] = annotated
    return collage


# @app.route('/detect', methods=['POST'])
# def detect():
#     """Detect WLD in an uploaded image."""
#     try:
#         # Get the uploaded image
#         image_file = request.files.get('file')
#         if not image_file:
#             return jsonify({"error": "No image uploaded"}), 400
#
#         # Read the image
#         image_bytes = image_file.read()
#         original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
#         if original_image is None:
#             return jsonify({"error": "Invalid image format"}), 400
#
#         # Start timer for processing
#         start_time = time.time()
#
#         image_copy = original_image.copy()
#         aggregated_detections = []
#         existing_boxes = []
#         cropped_images_base64 = []
#
#         # Dictionary to store model-specific detection counts and confidence
#         model_detections = {model_name: {"count": 0, "confidence": 0.0} for model_name in models.keys()}
#
#         # Process all models
#         results_by_model = {model_name: model(image_copy) for model_name, model in models.items()}
#
#         # Process detection results for each model
#         for model_name, results in results_by_model.items():
#             model_confidences = []
#             for result in results:
#                 for box in result.boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     conf = float(box.conf[0])
#                     class_id = int(box.cls[0])
#
#                     # Avoid duplicates and only include WLD detections
#                     if conf > 0.5 and class_id == 1 and not is_duplicate((x1, y1, x2, y2), existing_boxes):
#                         aggregated_detections.append((x1, y1, x2, y2, conf, class_id, model_name))
#                         existing_boxes.append((x1, y1, x2, y2))
#                         model_detections[model_name]["count"] += 1
#                         model_confidences.append(conf)
#
#                         # Crop and encode the detected area
#                         cropped_image = crop_image_with_padding(original_image, (x1, y1, x2, y2))
#                         _, buffer = cv2.imencode('.jpg', cropped_image)
#                         cropped_images_base64.append(base64.b64encode(buffer).decode('utf-8'))
#
#             # Calculate average confidence for this model
#             if model_confidences:
#                 model_detections[model_name]["confidence"] = sum(model_confidences) / len(model_confidences)
#
#         # Detection metadata
#         wld_detections = len(aggregated_detections)
#         detection_status = "No WLD Found" if wld_detections == 0 else "WLD Detected"
#         percentage_confidence = sum(d[4] for d in aggregated_detections) / len(
#             aggregated_detections) * 100 if aggregated_detections else 0
#
#         # End timer and calculate processing time
#         end_time = time.time()
#         processing_time = end_time - start_time
#
#         # Draw bounding boxes on a copy of the image
#         for x1, y1, x2, y2, conf, class_id, model_name in aggregated_detections:
#             color = (0, 0, 255)
#             cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 12)
#
#         # Create a collage image
#         collage_image = create_collage(original_image, image_copy)
#
#         # Save the original and collage images to files
#         original_image_path = 'path_to_save_original_image.jpg'
#         collage_image_path = 'path_to_save_collage_image.jpg'
#
#         cv2.imwrite(original_image_path, original_image)
#         cv2.imwrite(collage_image_path, collage_image)
#
#         # Encode the original and collage images to base64 by reading from saved files
#         with open(original_image_path, "rb") as image_file:
#             original_base64 = base64.b64encode(image_file.read()).decode("utf-8")
#
#         with open(collage_image_path, "rb") as image_file:
#             collage_base64 = base64.b64encode(image_file.read()).decode("utf-8")
#
#         # Return data to Flutter
#         return jsonify({
#             "message": "Detection completed",
#             "status": detection_status,
#             "confidence": f"{percentage_confidence:.2f}%",
#             "wld_count": wld_detections,
#             "processing_time": f"{processing_time:.2f} seconds",
#             "model_detections": model_detections,
#             "collage_image": collage_base64,
#             "cropped_images": cropped_images_base64,
#         })
#
#     except Exception as e:
#         return jsonify({"error": "An error occurred", "details": str(e)}), 500
@app.route('/detect', methods=['POST'])
def detect():
    """Detect WLD in an uploaded image with optimized and strict duplicate logic."""
    try:
        image_file = request.files.get('file')
        if not image_file:
            return jsonify({"error": "No image uploaded"}), 400

        image_bytes = image_file.read()
        original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if original_image is None:
            return jsonify({"error": "Invalid image format"}), 400

        start_time = time.time()
        image_copy = original_image.copy()
        aggregated_detections = []
        existing_boxes = []
        cropped_images_base64 = []

        model_detections = {model_name: {"count": 0, "confidence": 0.0} for model_name in models.keys()}

        results_by_model = {model_name: model(image_copy) for model_name, model in models.items()}

        for model_name, results in results_by_model.items():
            model_confidences = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])

                    if conf > 0.5 and class_id == 1 and not is_duplicate_strict((x1, y1, x2, y2), existing_boxes):
                        aggregated_detections.append((x1, y1, x2, y2, conf, class_id, model_name))
                        existing_boxes.append((x1, y1, x2, y2))
                        model_detections[model_name]["count"] += 1
                        model_confidences.append(conf)

                        cropped_image = crop_image_with_padding(original_image, (x1, y1, x2, y2))
                        _, buffer = cv2.imencode('.jpg', cropped_image)
                        cropped_images_base64.append(base64.b64encode(buffer).decode('utf-8'))

            if model_confidences:
                model_detections[model_name]["confidence"] = sum(model_confidences) / len(model_confidences)

        unique_detections = sorted(aggregated_detections, key=lambda x: x[4], reverse=True)

        wld_detections = len(unique_detections)
        detection_status = "No WLD Found" if wld_detections == 0 else "WLD Detected"
        percentage_confidence = sum(d[4] for d in unique_detections) / len(
            unique_detections) * 100 if unique_detections else 0

        end_time = time.time()
        processing_time = end_time - start_time

        for x1, y1, x2, y2, conf, class_id, model_name in unique_detections:
            color = (0, 0, 255)
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 12)

        collage_image = create_collage(original_image, image_copy)
        _, collage_buffer = cv2.imencode('.jpg', collage_image)
        collage_base64 = base64.b64encode(collage_buffer).decode("utf-8")

        return jsonify({
            "message": "Detection completed",
            "status": detection_status,
            "confidence": f"{percentage_confidence:.2f}%",
            "wld_count": wld_detections,
            "processing_time": f"{processing_time:.2f} seconds",
            "model_detections": model_detections,
            "collage_image": collage_base64,
            "cropped_images": cropped_images_base64,
        })

    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
