import cv2
import rasterio

def annotate_image(file_path, detections):
    with rasterio.open(file_path) as src:
        image = src.read()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    annotated_path = f"results/annotated_{os.path.basename(file_path)}"
    cv2.imwrite(annotated_path, image)
    return annotated_path
