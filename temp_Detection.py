import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision import Detections, BoxAnnotator

class ObjectDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

        # Initialize ColorPalette directly
        self.palette = ColorPalette(colors=['#FF0000', '#00FF00', '#0000FF'])
        self.box_annotator = BoxAnnotator(color=self.palette, thickness=3)

    def load_model(self):
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        # Setup detections for visualization
        detections = Detections(
           xyxy=results[0].boxes.xyxy.cpu().numpy(),
           confidence=results[0].boxes.conf.cpu().numpy(),
           class_id=results[0].boxes.cls.cpu().numpy().astype(int),
       )

        # Format custom labels
        labels = [
            f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id)
        ]

        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections)

        # Draw labels separately
        for i, (xyxy, label) in enumerate(zip(detections.xyxy, labels)):
            x1, y1, x2, y2 = xyxy
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame
    
    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()
