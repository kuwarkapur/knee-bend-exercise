import streamlit as st
import av
import numpy as np
import mediapipe as mp
import cv2
from streamlit_webrtc import (
    ClientSettings,
    WebRtcMode,
    RTCConfiguration,
    webrtc_streamer,
    VideoTransformerBase,
)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
st.title("Knee Bend Exercise Demo")
video_file = open(r'C:\Users\Kuwar\Desktop\Recording 2023-03-27 160441.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)


class KneeBendDetector(VideoTransformerBase):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.angle_threshold = 170
        
    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        image = frame.to_ndarray(format="bgr24")

        with self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = {
                    "nose": results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE],
                    "left_hip": results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP],
                    "right_hip": results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP],
                    "left_knee": results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE],
                    "right_knee": results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE],
                }

                left_angle = self.calculate_angle(
                    landmarks["left_hip"], landmarks["left_knee"], landmarks["nose"]
                )
                right_angle = self.calculate_angle(
                    landmarks["right_hip"], landmarks["right_knee"], landmarks["nose"]
                )

                # Determine if exercise is performed correctly
                if left_angle > self.angle_threshold and right_angle > self.angle_threshold:
                    status = "Good"
                else:
                    status = "Bad"

                # Draw angle values on frame
                cv2.putText(
                    image,
                    f"Left angle: {round(left_angle, 2)} degrees",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"Right angle: {round(right_angle, 2)} degrees",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"Status: {status}",
                    (10,150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                    )
        return image
    


def main():
    st.title("Knee Bend Exercise Detector")
    #client_settings = ClientSettings(
     #   rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
      #  media_stream_constraints={"video": True, "audio": False},
       # video_transformer_factory=KneeBendDetector,
       # webrtc_mode=WebRtcMode.SENDRECV,
    #)
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV,rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=KneeBendDetector)
    
    
if __name__ == "__main__":
    main()
    
