from xml.parsers.expat import model
import cv2
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker, CourtTracker
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models


def main():
    # read video
    input_video_path = "input_videos/long_ralley.mov"
    video_frames, fps = read_video(input_video_path)

    # detect players
    player_tracker = PlayerTracker('models/pickleball_better.pt')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")

    # detect balls
    ball_tracker = BallTracker("models/pickleball_better.pt")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # detect courts
    court_tracker = CourtTracker("models/court_detection.pt")
    court_detections = court_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/court_detections.pkl")

    # draw player bboxes
    output_video_frames = player_tracker.draw_boxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_boxes(output_video_frames, ball_detections)
    output_video_frames = court_tracker.draw_boxes(output_video_frames, court_detections)

    # detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(ball_shot_frames)
    frames_for_hit = 20
    curr_shot = None
    
    # technique classification
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize for ResNet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])
    

    # Load the image
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)  # 4 classes: forehand/backhand/ready_position/serve

    model.load_state_dict(torch.load("models/resnet18_technique.pth", map_location=torch.device('cpu')))
    model.eval()  # set to inference mode
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    currPrediction = None

    # draw frame number in top left corner
    for i, frame in enumerate(output_video_frames):
        if i in ball_shot_frames:
            curr_shot = i
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert NumPy array to PIL Image
            image = Image.fromarray(frame_rgb)
            # Apply transform, convert image to PyTorch tensor
            image_tensor = transform(image).unsqueeze(0).to(device)  # type: ignore

            # Inference
            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor)
                _, pred = torch.max(outputs, 1)

            print("Predicted class index:", pred.item())
            classes = ['backhand', 'forehand', 'ready_position', 'serve']
            print("Predicted label:", classes[int(pred.item())])
            currPrediction = classes[int(pred.item())]

            # run it through the technique classification model
        if curr_shot is not None and i < curr_shot + frames_for_hit:
            cv2.putText(frame, "Ball Hit!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            if currPrediction is not None:
                cv2.putText(frame, f"Technique: {currPrediction}", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
            
        cv2.putText(frame, f"Frame: {i}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    print(f"Video FPS: {fps}")
    save_video(output_video_frames, "output_videos/processed_vid.avi",fps)

if __name__ == "__main__":
    main()