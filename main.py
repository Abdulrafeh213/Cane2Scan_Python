import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog, Label, Button, Canvas, ttk, Frame, Toplevel
from PIL import Image, ImageTk
import numpy as np
from firebase_admin import credentials, firestore, initialize_app
from datetime import datetime
from tkinter import PhotoImage
import time

# Firebase setup
cred = credentials.Certificate('D:/FYP project/cane2scan_python/database/project-fyp-7e8c1-firebase-adminsdk-ehwmg-d03dca4472.json')
initialize_app(cred)
db = firestore.client()

# YOLO models
models = {
    "yolov8s": YOLO('D:/FYP project/cane2scan_python/trained/yolov8s/weights/best.pt'),
    "yolov8n": YOLO('D:/FYP project/cane2scan_python/trained/yolov8n/weights/best.pt'),
    "yolov8m": YOLO('D:/FYP project/cane2scan_python/trained/yolov8m/weights/best.pt'),
    "yolov8l": YOLO('D:/FYP project/cane2scan_python/trained/yolov8l/weights/best.pt'),
}

class_labels = {0: 'Healthy Leaf', 1: 'WLD'}

def is_duplicate(box, existing_boxes, threshold=0.5):
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


def draw_bounding_box_cv2(image, detections, padding=10):
    cropped_images = []
    for detection in detections:
        x1, y1, x2, y2, conf, class_id, model_name = detection  # Include model_name
        # Add 10px padding around the bounding box
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        color = (0, 0, 255)
        if int(class_id) == 1:
            thickness = 20
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            text = f"{class_labels[int(class_id)]} ({conf * 100:.1f}%)\n{model_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = color
            thickness_text = 2
            text_size = cv2.getTextSize(text, font, 0.6, thickness_text)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (255, 255, 255),
                          cv2.FILLED)
            cv2.putText(image, text, (text_x, text_y), font, 0.6, text_color, thickness_text)
            cropped_images.append(image[y1:y2, x1:x2])
    return image, cropped_images


def display_result_image(canvas, image_path):
    image = Image.open(image_path)
    image = image.resize((1000, 500), Image.Resampling.LANCZOS)  # Resize the result image to fit the canvas size
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(500, 250, anchor="center", image=photo)  # Place image at the center of the canvas
    canvas.image = photo





def browse_image(root, canvas, result_label, slider_frame, slider_canvas, scrollbar):
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if image_path:
        # Display progress.gif while processing the image
        progress_image = PhotoImage(file="progres.gif")
        canvas.create_image(500, 250, anchor="center", image=progress_image)
        canvas.image = progress_image
        result_label.config(text="Please wait, your image is being processed")
        root.update()

        # Process the image and detect WLD
        detect_and_visualize(image_path, root, canvas, result_label, slider_frame, slider_canvas, scrollbar)


def detect_and_visualize(image_path, root, canvas, result_label, slider_frame, slider_canvas, scrollbar):
    original_image = cv2.imread(image_path)
    if original_image is None:
        result_label.config(text="Error: Unable to load the image.")
        return

    # Start timer to measure processing time
    start_time = time.time()

    image_copy = original_image.copy()
    aggregated_detections = []
    existing_boxes = []

    # Dictionary to store model-specific detection counts
    model_detections = {model_name: 0 for model_name in models.keys()}

    # Process all models in parallel
    results_by_model = {model_name: model(image_copy) for model_name, model in models.items()}

    # Iterate through each model and its results
    for model_name, results in results_by_model.items():
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                # Avoid duplicates and only include WLD detections
                if conf > 0.5 and class_id == 1 and not is_duplicate((x1, y1, x2, y2), existing_boxes):
                    aggregated_detections.append((x1, y1, x2, y2, conf, class_id, model_name))  # Include model name
                    existing_boxes.append((x1, y1, x2, y2))
                    model_detections[model_name] += 1  # Increment model-specific detection count

    # Draw bounding boxes and generate cropped images
    image_with_boxes, cropped_images = draw_bounding_box_cv2(image_copy, aggregated_detections, padding=10)
    wld_detections = len(aggregated_detections)

    # Combine original and annotated images
    combined_image = np.hstack((original_image, image_with_boxes))
    combined_image_path = "result_collage.jpg"
    cv2.imwrite(combined_image_path, combined_image)

    # Detection status and confidence percentage
    detection_status = "No WLD Found" if wld_detections == 0 else "WLD Detected"
    percentage_confidence = sum(d[4] for d in aggregated_detections) / len(aggregated_detections) * 100 if aggregated_detections else 0

    # End timer and calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time

    # Update result label to show detection status, confidence, and processing time
    result_label.config(
        text=f"{detection_status}\nConfidence: {percentage_confidence:.2f}%\nWLD Found: {wld_detections}\nProcessing Time: {processing_time:.2f} seconds"
    )

    # Update result label to show model-specific detection counts
    for model_name, count in model_detections.items():
        result_label.config(
            text=f"{result_label.cget('text')}\n{model_name.upper()} detected {count} WLD(s)."
        )

    # Log results in Firebase
    db.collection("detections").add({
        "status": detection_status,
        "confidence": f"{percentage_confidence:.2f}%",
        "processing_time": f"{processing_time:.2f} seconds",
        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Display the combined result image on the canvas
    display_result_image(canvas, combined_image_path)

    # Update the slider with cropped images
    for widget in slider_canvas.winfo_children():
        widget.destroy()

    for i, cropped in enumerate(cropped_images):
        cropped_filename = f"cropped_{i + 1}.jpg"
        cv2.imwrite(cropped_filename, cropped)
        cropped_image_display = Image.open(cropped_filename)
        cropped_image_display = cropped_image_display.resize((150, 150), Image.Resampling.LANCZOS)
        cropped_photo = ImageTk.PhotoImage(cropped_image_display)
        cropped_label = Label(slider_canvas, image=cropped_photo)
        cropped_label.image = cropped_photo
        cropped_label.pack(pady=5)

        # Click event to open the cropped image in a new window
        def on_cropped_image_click(cropped_filename=cropped_filename):
            open_cropped_image(cropped_filename)

        cropped_label.bind("<Button-1>", lambda e, cropped_filename=cropped_filename: on_cropped_image_click(cropped_filename))

    # Update the scrollable region for the canvas
    slider_canvas.config(scrollregion=slider_canvas.bbox("all"))

    if len(cropped_images) > 4:
        scrollbar.pack(side="right", fill="y")
        slider_canvas.config(yscrollcommand=scrollbar.set)
    else:
        scrollbar.pack_forget()




def open_cropped_image(cropped_filename):
    # Create a new window to display the cropped image
    cropped_image = Image.open(cropped_filename)
    cropped_image.show()


def main():
    root = Tk()
    root.title("Cane2Scan")
    root.geometry("1000x700")
    root.configure(bg="#e0f7fa")

    # Set fullscreen mode
    root.attributes('-fullscreen', True)

    # Title
    title_label = Label(root, text="Cane2Scan: White Leaf Disesae Detection System", font=("Arial", 40, "bold"), fg="#00695c", bg="#e0f7fa")
    title_label.pack(pady=10)

    main_frame = Frame(root, bg="#e0f7fa")
    main_frame.pack(fill="both", expand=True)

    # Canvas for slider with scroll functionality
    slider_frame = Frame(main_frame,  width=200)
    slider_frame.pack(side="left", fill="x", padx=10)

    slider_canvas = Canvas(slider_frame)
    slider_canvas.pack(side="left", fill="both", expand=True)

    # Add a scrollbar to the slider canvas
    scrollbar = ttk.Scrollbar(slider_frame, orient="vertical", command=slider_canvas.xview)
    scrollbar.pack(side="right", fill="x")
    slider_canvas.configure(xscrollcommand=scrollbar.set)

    canvas = Canvas(main_frame, width=1000, height=500)
    canvas.pack(pady=20)

    # Load image by default (replace 'your_image.png' with your image file path)
    image = PhotoImage(file="logo1.png")
    canvas.create_image(500, 250, anchor="center", image=image)
    canvas.image = image


    result_label = Label(main_frame, text="Please upload an image", bg="#e0f7fa", font=("Arial", 14))
    result_label.pack()

    browse_button = Button(main_frame, text="Browse Image", command=lambda: browse_image(root, canvas, result_label, slider_frame, slider_canvas, scrollbar), bg="#00796b", fg="white", font=("Arial", 12))
    browse_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
