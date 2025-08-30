

# class PoseEstimator:
#     def __init__(self, mode=False, upperBody=False, smoothLandmarks=True, 
#                  detectionConfidence=0.5, trackingConfidence=0.5) -> None:
        
#         self.mode = mode
#         self.upperBody = upperBody
#         self.smoothLandmarks = smoothLandmarks
#         self.detectionConfidence = detectionConfidence
#         self.trackingConfidence = trackingConfidence
#         self.mpDraw = mp.python.solutions.drawing_utils
#         self.mpPose = mp.python.solutions.pose
#         self.pose = self.mpPose.Pose(self.mode, self.upperBody, self.smoothLandmarks,
#                                       self.detectionConfidence, self.trackingConfidence)

#     def findPose(self, image, draw=True):
#         imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = self.pose.process(imageRGB)
#         if results.pose_landmarks:
#             if draw:
#                 self.mpDraw.draw_landmarks(image, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
#         return image