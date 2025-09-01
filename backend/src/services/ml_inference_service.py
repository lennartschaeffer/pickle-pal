import os
import cv2
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker, CourtTracker
from pose_detection import PoseClassifier
from data import MLInferenceResult

class MLInferenceService:
    def __init__(self, input_video_path : str):
        self.input_video_path = input_video_path
        self.player_tracker = PlayerTracker('ml/models/pickleball_better.pt')
        self.ball_tracker = BallTracker("ml/models/pickleball_better.pt")
        self.court_tracker = CourtTracker("ml/models/court_detection.pt")
        self.pose_classifier = PoseClassifier(
            pose_model_path='ml/models/pose_landmarker_lite.task',
            classifier_path='ml/models/new_pose_classifier.pth'
        )

    def predict(self, data) -> MLInferenceResult:
        video_frames, fps = read_video(self.input_video_path)

        player_detections = self.player_tracker.detect_frames(video_frames)
        
        ball_detections = self.ball_tracker.detect_frames(video_frames)
        ball_detections = self.ball_tracker.interpolate_ball_positions(ball_detections)
        
        court_detections = self.court_tracker.detect_frames(video_frames)
        
        # draw player bboxes
        output_video_frames = self.player_tracker.draw_boxes(video_frames, player_detections)
        output_video_frames = self.ball_tracker.draw_boxes(output_video_frames, ball_detections)
        output_video_frames = self.court_tracker.draw_boxes(output_video_frames, court_detections)

        # detect ball shots
        ball_shot_frames = self.ball_tracker.get_ball_shot_frames(ball_detections)
        frames_for_hit = 50
        curr_shot = None
        ball_hit_count = 0
        ralley_over_frames = 100
        frame_ball_last_hit = None
        fh_bh_counts = {}
        
        # draw frame number in top left corner
        for i, frame in enumerate(output_video_frames):
            if i in ball_shot_frames:
                curr_shot = i
                ball_hit_count += 1
                frame_ball_last_hit = i
            if curr_shot is not None:
                if i < curr_shot + frames_for_hit:
                    cv2.putText(frame, "Ball Hit!", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    result, _ = self.pose_classifier.classify_pose(frame)
                    fh_bh_counts[result] = fh_bh_counts.get(result, 0) + 1
            if frame_ball_last_hit is not None and i > frame_ball_last_hit + ralley_over_frames:
                curr_shot = None
                frame_ball_last_hit = None
                cv2.putText(frame, f"Rally Over! Total Hits: {ball_hit_count}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.putText(frame, f"Total Hits: {ball_hit_count}", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Frame: {i}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            
        print(fh_bh_counts, ball_hit_count)

        forehands = fh_bh_counts.get("Forehand", 0)
        backhands = fh_bh_counts.get("Backhand", 0)
        total_hits = forehands + backhands
        
        output_dir = os.path.dirname("../output_videos/processed_vid.avi")
        os.makedirs(output_dir, exist_ok=True)
    
        # get forehand/backhand percentage
        save_video(output_video_frames, "../output_videos/processed_vid.avi",fps)

        return MLInferenceResult(
            technique_counts=fh_bh_counts,
            ralley_length=ball_hit_count,
            forehand_percentage=forehands / total_hits * 100 if total_hits > 0 else 0.0,
            backhand_percentage=backhands / total_hits * 100 if total_hits > 0 else 0.0
        )