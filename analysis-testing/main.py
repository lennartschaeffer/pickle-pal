import cv2
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker, CourtTracker
from pose_detection import PoseClassifier

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
    frames_for_hit = 50
    curr_shot = None
    ball__hit_count = 0
    ralley_over_frames = 100
    frame_ball_last_hit = None
    fh_bh_counts = {}
    
    # pose classfier
    pose_classifier = PoseClassifier(
        pose_model_path='models/pose_landmarker_lite.task',
        classifier_path='models/new_pose_classifier.pth'
    )
    
    # draw frame number in top left corner
    for i, frame in enumerate(output_video_frames):
        if i in ball_shot_frames:
            curr_shot = i
            ball__hit_count += 1
            frame_ball_last_hit = i
        if curr_shot is not None:
            if i < curr_shot + frames_for_hit:
                cv2.putText(frame, "Ball Hit!", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                result, _ = pose_classifier.classify_pose(frame)
                fh_bh_counts[result] = fh_bh_counts.get(result, 0) + 1
                max_technique = ("None", 0)
                for t in fh_bh_counts:
                    if fh_bh_counts[t] > max_technique[1]:
                        max_technique = (t, fh_bh_counts[t])
                cv2.putText(frame, f"P1 Current Technique: {max_technique[0]}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif i == curr_shot + frames_for_hit:
                print(fh_bh_counts)
                final_guess = max_technique[0]
                cv2.putText(frame, f"P1 Final Technique: {final_guess}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                fh_bh_counts = {}
        if frame_ball_last_hit is not None and i > frame_ball_last_hit + ralley_over_frames:
            curr_shot = None
            frame_ball_last_hit = None
            cv2.putText(frame, f"Rally Over! Total Hits: {ball__hit_count}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(frame, f"Total Hits: {ball__hit_count}", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Frame: {i}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    save_video(output_video_frames, "output_videos/processed_vid.avi",fps)

if __name__ == "__main__":
    main()