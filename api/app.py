import cv2 
from flask import Flask,Response
import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose

# Drawing all landmarks
mp_drawing = mp.solutions.drawing_utils

# Calculate angle
def calculate_angle(a, b, c):
    a = np.array(a) # hip
    b = np.array(b) # knee
    c = np.array(c) # ankle
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle    
    return angle 

app=Flask(__name__)

@app.route('/')
def index():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def frames():
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    relax = 0 
    bent = 0
    counter = 0
    feedback=None
    stage=None
    images_arr=[]

    # Mediapipe script
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # detection
            results = pose.process(image)
        
            # RGB to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                
                # Calculating Angle
                angle = calculate_angle(hip, knee, ankle)

                # Render Detections
                a0 = int(ankle[0] * width)
                a1 = int(ankle[1] * height)

                k0 = int(knee[0] * width)
                k1 = int(knee[1] * height)

                h0 = int(hip[0] * width)
                h1 = int(hip[1] * height)

                cv2.line(image, (h0, h1), (k0, k1), (255, 255, 0), 2)
                cv2.line(image, (k0, k1), (a0, a1), (255, 255, 0), 2)
                cv2.circle(image, (h0, h1), 5, (0, 0, 0))
                cv2.circle(image, (k0, k1), 5, (0, 0, 0))
                cv2.circle(image, (a0, a1), 5, (0, 0, 0))       
                
                # For Visualizing Angle
                cv2.putText(image, str(round(angle,4)), tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                relax_time = (1 / fps) * relax
                bent_time = (1 / fps) * bent

                #Counter Logic
                if angle > 140:
                    relax += 1
                    bent = 0
                    stage = "Relaxed"
                
                elif angle < 140:
                    relax = 0
                    bent += 1
                    stage = "Bent"
                
                # complete rep
                if bent_time == 8:
                    counter += 1
                elif bent_time < 8 and stage == 'Bent':
                    feedback="Keep your knee bent"
                else:
                    feedback = f"{counter} reps done"
                    
            except:
                pass
                    
            # stats screen
            cv2.rectangle(image, (0,0), (int(height/3.5), 400), (0,0,255), -1)
            
            # total stats
            cv2.putText(image, 'STATS', (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
            
            # how many reps
            cv2.putText(image, 'Reps', (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (15,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage of exercise
            cv2.putText(image, 'Position', (15,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (15,150),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Feedback
            cv2.putText(image, 'FeeDback', (15,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, feedback, (15,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Bent Time
            cv2.putText(image, 'Bent-Time', (15,300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(bent_time,2)), (15,350),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)  

        
        
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

if '__main__'==__name__:
    app.run(debug=True)

