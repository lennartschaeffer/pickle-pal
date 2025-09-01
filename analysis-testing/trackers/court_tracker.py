from shapely import Polygon
from ultralytics import YOLO
import pickle
import cv2

class CourtTracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect_frames(self,frames, read_from_stub=False,stub_path=None) -> list:
        court_detections = []
        
        # reuse cached results
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                court_detections = pickle.load(f)
            return court_detections
        
        for frame in frames:
            court_dict = self.detect_frame(frame)
            court_detections.append(court_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(court_detections, f)

        return court_detections

    def detect_frame(self,frame) -> dict:
        results = self.model.predict(frame)[0]
        court_dict = {}

        if results.masks:
            for mask in results.masks:
                # Take the first mask as the main court
                mask = mask.xy[0]
                pts = mask.astype(int)
                court_dict[1] = pts
        
        return court_dict
            
    def draw_boxes(self, video_frames, court_detections) -> list:
        output_video_frames = []
        for frame, court_dict in zip(video_frames, court_detections):
            # draw boxes
            for track_id, pts in court_dict.items():
                cv2.polylines(frame, [pts], isClosed=True, color=(0,255,0), thickness=2)
                cv2.putText(frame, f"Court {track_id}", (int(pts[0][0]), int(pts[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            output_video_frames.append(frame)
            
        return output_video_frames