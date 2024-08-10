import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
#from supervision.tools.detections import Detections, BoxAnnotator
from supervision import Detections, BoxAnnotator

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from sort import Sort
from deep_sort_realtime.deepsort_tracker import DeepSort

SHOW_ONLY_CLASS_ID:int = 24
BBOX_SORT_ESTIMATE_DETECTION_BBOX:float = 5.0


class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names

        color=ColorPalette(colors=['#FF0000', '#00FF00', '#0000FF'])
        
        self.box_annotator = BoxAnnotator(color=color, thickness=3)
    

    def load_model(self):
       
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def get_results(self, results):
        
        xyxys = []
        confidences = []
        class_ids = []
        detections_list = []
        
        # Extract detections for person class
        for result in results[0]:
            class_id_arr = result.boxes.cls.cpu().numpy()
            class_id=class_id_arr.astype(int)
            
            if class_id == SHOW_ONLY_CLASS_ID:
                
                bbox = result.boxes.xyxy.cpu().numpy()
                confidence = result.boxes.conf.cpu().numpy()
                
                
                merged_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], confidence[0], class_id_arr[0]]
                
                
                detections_list.append(merged_detection)
            #xyxys.append(bbox)
            #confidences.append(confidence)
            #class_ids.append(class_id)
                
    
        return np.array(detections_list)
    
    
    def draw_bounding_boxes(self, img, bboxes, ids, class_ids, confidences):
  
        for bbox, id_, class_id, confidence in zip(bboxes, ids, class_ids, confidences):
            cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
            cv2.putText(img, "ID: " + str(id_) + " " + str(self.CLASS_NAMES_DICT[int(class_id)]) + " " + str(round(confidences[0], 2)), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img
    
    
        
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # SORT
        sort = Sort(max_age=30, min_hits=5, iou_threshold=0.3)
        
        while True:
        
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            detections_list = self.get_results(results)
            
            # SORT Tracking
            if len(detections_list) == 0:
                detections_list = np.empty((0, 6))
        
            res = sort.update(detections_list)
                
            boxes_track = res[:,:-1]
            boxes_ids = res[:,-1].astype(int)
            
            # Check if boxes_ids is not empty and add alert
            if len(boxes_ids) > 0:
                alert_text = "ALERT: Object detected!"
                cv2.putText(frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        

            # Match boxes_track with detections_list to find confidence[0] and class_id_arr[0]
            confidences = []
            class_ids = []
            for box in boxes_track:
                for detection in detections_list:
                    if np.allclose(box[:4], detection[:4], atol=BBOX_SORT_ESTIMATE_DETECTION_BBOX):  # Compare with tolerance
                        confidence = detection[4]
                        confidences.append(confidence)
                        class_id = int(detection[5])
                        class_ids.append(class_id)


            frame = self.draw_bounding_boxes(frame, boxes_track, boxes_ids, class_ids, confidences)
            
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


