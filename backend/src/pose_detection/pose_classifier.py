import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
    
class PoseClassifierModel(nn.Module):
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

class PoseClassifier:
    def __init__(self, pose_model_path, classifier_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["Backhand", "Forehand"]

        # custom pose classifier
        self.model = PoseClassifierModel(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.model.eval()

        # mediapipe pose landmarker
        base_options = python.BaseOptions(model_asset_path=pose_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # needed landmarks for my pose classifier
        self.needed_landmarks_indices = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]

    def extract_landmarks(self, pose_landmarks):
        left_shoulder = None
        right_shoulder = None
        mapped_landmarks = []
        for i, landmark in enumerate(pose_landmarks):
            if i in self.needed_landmarks_indices:
                mapped_landmarks.append((landmark.x, landmark.y))
            if i == 11:
                left_shoulder = (landmark.x, landmark.y)
            if i == 12:
                right_shoulder = (landmark.x, landmark.y)
        if left_shoulder and right_shoulder:
            neck = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
            mapped_landmarks.append((neck[0], neck[1]))
        return mapped_landmarks

    def classify_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        results = []
        prediction, probs = None, None
        for pose_landmarks in detection_result.pose_landmarks:
            mapped_landmarks = self.extract_landmarks(pose_landmarks)
            converted_landmarks = []
            img_width = 1280
            img_height = 720
            for x, y in mapped_landmarks:
                converted_x = int(x * img_width)
                converted_y = int(y * img_height)
                converted_landmarks.append((converted_x, converted_y))
            x = np.array(converted_landmarks).flatten()
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(x_tensor)
                predicted_class = pred.softmax(dim=1)
            prediction = self.class_names[int(predicted_class.argmax())]
            probs = predicted_class.cpu().numpy()
            # results.append({
            #     "class_probs": predicted_class.cpu().numpy(),
            #     "class_name": self.class_names[int(predicted_class.argmax())]
            # })
        return prediction, probs

    def detect_frames(self, frames):
        all_results = []
        for frame in frames:
            results, _ = self.classify_pose(frame)
            all_results.append(results)
        return all_results

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # type: ignore
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks # type: ignore
            ])
            solutions.drawing_utils.draw_landmarks( # type: ignore
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS, # type: ignore
                solutions.drawing_styles.get_default_pose_landmarks_style()) # type: ignore
        return annotated_image