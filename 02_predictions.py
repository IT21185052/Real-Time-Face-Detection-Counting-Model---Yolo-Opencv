import cv2
from yolo_predictions import YOLO_Pred  # Ensure this module is correctly implemented

# Load YOLO model
yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')

# Read and predict on a single image
img = cv2.imread('street_image.jpg')
if img is None:
    print("Error: Could not read image file.")
else:
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)  # Wait for key press before closing

    img_pred = yolo.predictions(img)
    cv2.imshow('YOLO Prediction', img_pred)
    cv2.waitKey(0)

cv2.destroyAllWindows()  # Close all OpenCV windows

# Process video file
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or error reading frame.")
            break
        
        pred_image = yolo.predictions(frame)
        cv2.imshow('YOLO Video Prediction', pred_image)

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

    cap.release()

cv2.destroyAllWindows()  # Ensure all windows are closed
