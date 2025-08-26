import cv2
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker, CourtTracker

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
    # draw frame number in top left corner
    for i, frame in enumerate(output_video_frames):
        if i in ball_shot_frames:
            curr_shot = i
        if curr_shot is not None and i < curr_shot + frames_for_hit:
            cv2.putText(frame, "Ball Hit!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        cv2.putText(frame, f"Frame: {i}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    print(f"Video FPS: {fps}")
    save_video(output_video_frames, "output_videos/processed_vid.avi",fps)

if __name__ == "__main__":
    main()