import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred:
    def __init__(self, onnx_model, data_yaml):
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect_faces(self, frame):
        row, col, _ = frame.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = frame
        
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()
        
        detections = preds[0]
        boxes, confidences, classes = [], [], []
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO
        
        for row in detections:
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                if class_score > 0.25:
                    cx, cy, w, h = row[:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    
                    boxes.append([left, top, width, height])
                    confidences.append(confidence)
                    classes.append(class_id)
        
        boxes_np, confidences_np = np.array(boxes), np.array(confidences)
        indexes = cv2.dnn.NMSBoxes(boxes_np.tolist(), confidences_np.tolist(), 0.25, 0.45)
        
        if isinstance(indexes, tuple) or len(indexes) == 0:
            indexes = []
        else:
            indexes = indexes.flatten()
        
        for i in indexes:
            x, y, w, h = boxes_np[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return frame, len(indexes)

def real_time_face_detection():
    yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        pred_frame, face_count = yolo.detect_faces(frame)
        cv2.putText(pred_frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-Time Face Detection", pred_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_face_detection()












'''

import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred:
    def __init__(self, onnx_model, data_yaml):
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect_faces(self, frame):
        row, col, _ = frame.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = frame
        
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()
        
        detections = preds[0]
        boxes, confidences, classes = [], [], []
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO
        
        for row in detections:
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                if class_score > 0.25:
                    cx, cy, w, h = row[:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    
                    boxes.append([left, top, width, height])
                    confidences.append(confidence)
                    classes.append(class_id)
        
        boxes_np, confidences_np = np.array(boxes), np.array(confidences)
        indexes = cv2.dnn.NMSBoxes(boxes_np.tolist(), confidences_np.tolist(), 0.25, 0.45)
        
        if isinstance(indexes, tuple) or len(indexes) == 0:
            indexes = []
        else:
            indexes = indexes.flatten()
        
        for i in indexes:
            x, y, w, h = boxes_np[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return frame, len(indexes)

def real_time_face_detection():
    yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    high_face_count = 0  # Counter for consecutive frames with 3 or more faces
    high_face_threshold = 10  # Number of frames before shutting down

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        pred_frame, face_count = yolo.detect_faces(frame)

        # Check if detected faces are 3 or more
        if face_count >= 3:
            high_face_count += 1
        else:
            high_face_count = 0  # Reset if faces are less than 3

        cv2.putText(pred_frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-Time Face Detection", pred_frame)

        # If 3 or more faces detected for `high_face_threshold` consecutive frames, shut down
        if high_face_count >= high_face_threshold:
            print("3 or more faces detected for multiple frames. Shutting down camera.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_face_detection()

'''