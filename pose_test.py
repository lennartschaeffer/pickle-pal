from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn

class PoseClassifier(nn.Module):
    def __init__(self, num_classes=2, num_keypoints=18):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_keypoints * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseClassifier(num_classes=2).to(device)
model.load_state_dict(torch.load("models/new_pose_classifier.pth", map_location=device))

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList() # type: ignore
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks # type: ignore
    ])
    solutions.drawing_utils.draw_landmarks( # type: ignore
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS, # type: ignore
      solutions.drawing_styles.get_default_pose_landmarks_style()) # type: ignore
  return annotated_image

# Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

video_path = "input_videos/vid.mov"
cap = cv2.VideoCapture(video_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    needed_landmarks_indices = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]
    # Convert BGR frame (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)
    for pose_landmarks in detection_result.pose_landmarks:
        left_shoulder = None
        right_shoulder = None
        mapped_landmarks = []
        for i, landmark in enumerate(pose_landmarks):
            if i in needed_landmarks_indices:
                mapped_landmarks.append((landmark.x, landmark.y))
            if i == 11:
                left_shoulder = (landmark.x, landmark.y)
            if i == 12:
                right_shoulder = (landmark.x, landmark.y)
        if left_shoulder and right_shoulder:
            neck = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
            mapped_landmarks.append((neck[0], neck[1]))
            
        converted_landmarks = []
        img_width = 1280
        img_height = 720
        for x,y in mapped_landmarks:
            converted_x = int(x * img_width)
            converted_y = int(y * img_height)
            converted_landmarks.append((converted_x, converted_y))
        x = np.array(converted_landmarks).flatten()  # shape (36,)
        
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # shape (1,36)
        x_tensor = x_tensor.to(device) # Move tensor to the same device as the model

        model.eval()
        with torch.no_grad():
            pred = model(x_tensor)
            predicted_class = pred.softmax(dim=1)

        print(predicted_class)
        class_names = ["Backhand", "Forehand"]
        print("Predicted class:", class_names[int(predicted_class.argmax())])

    # STEP 5: Process the detection result. Visualize it.
    annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
    cv2.imshow("Annotated Frame", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()