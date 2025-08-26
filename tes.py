import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

# load YOLO models
player_ball_model = YOLO("models/pickleball_better.pt")  # players + ball
court_model = YOLO("models/seg_plz.pt")  # court segmentation

video = "input_videos/vid.mov"
cap = cv2.VideoCapture(video)

# Rally tracking state
rally_length = 0
prev_center = None
prev_direction = None
court_polygon = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    court_results = court_model(frame)
    for result in court_results:
        if result.masks is not None:
            # Take the first mask as the main court
            mask = result.masks.xy[0]
            court_polygon = Polygon(mask)  # shapely polygon for in/out test
            pts = mask.astype(int)
            cv2.polylines(frame, [pts], isClosed=True, color=(0,255,0), thickness=2)
    
    player_results = player_ball_model(frame)

    ball_center = None
    for result in player_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # If this is the ball, store its center
            if cls_name.lower() == "ball":
                ball_center = ((x1 + x2)//2, (y1 + y2)//2)
                cv2.circle(frame, ball_center, 5, (0,0,255), -1)

    # ralley length
    if ball_center is not None and court_polygon is not None:
        ball_point = Point(ball_center)

        # If ball goes out of bounds → rally ends
        if not court_polygon.contains(ball_point):
            print(f"Rally ended, length: {rally_length}")
            rally_length = 0
            prev_center = None
            prev_direction = None
        else:
            # Check direction change
            if prev_center is not None:
                dx = ball_center[0] - prev_center[0]
                dy = ball_center[1] - prev_center[1]
                direction = np.arctan2(dy, dx)
                
                if prev_direction is None or abs(direction - prev_direction) > np.pi/4:
                    rally_length += 1
                    
                    print(f"Hit detected, rally length = {rally_length}")
                    prev_direction = direction
                    
            cv2.putText(frame, f"Rally Length: {rally_length}", (10,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            prev_center = ball_center

    cv2.imshow("Video Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
