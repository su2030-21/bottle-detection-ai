

# ==========================================
# 🔥 VS CODE SYSTEM (BEST BOTTLE COUNTING)
# YOLOv8 SEGMENTATION + OpenCV
# ==========================================

import os
import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog

# Folders
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "processed_images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 🔥 BEST MODEL
model = YOLO("yolov8m-seg.pt")


# ==========================================
# 🧹 CLEAR OLD FILES
# ==========================================

def clear_folder(folder):
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))


# ==========================================
# 📤 SELECT IMAGE (FILE PICKER)
# ==========================================

def select_image():
    root = Tk()
    root.withdraw()  # hide main window

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        print("❌ No file selected")
        return None

    return file_path


# ==========================================
# 🤖 DETECT & COUNT BOTTLES
# ==========================================

def detect_bottles(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Error reading image")
        return

    # Resize
    image = cv2.resize(image, None, fx=1.2, fy=1.2)

    # 🔥 HIGH ACCURACY SETTINGS
    results = model.predict(
        source=image,
        conf=0.35,
        iou=0.5,
        max_det=300,
        verbose=False
    )

    boxes = results[0].boxes
    count = 0

    # Count bottles
    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label.lower() == "bottle":
            count += 1

    # Annotated image
    annotated = results[0].plot()

    # Save result
    output_path = os.path.join(OUTPUT_FOLDER, "result.jpg")
    cv2.imwrite(output_path, annotated)

    # Show image (OpenCV window)
    cv2.imshow("Bottle Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("================================")
    print(f"✅ TOTAL BOTTLES: {count}")
    print(f"📁 Saved at: {output_path}")
    print("================================")


# ==========================================
# ▶️ MAIN
# ==========================================

def main():
    print("Select an image...")

    image_path = select_image()

    if image_path:
        detect_bottles(image_path)


if __name__ == "__main__":
    main()
