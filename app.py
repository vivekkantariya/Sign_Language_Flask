from flask import Flask,render_template,Response
from keypoint import Keypoint
import cv2
import pandas as pd
import mediapipe as mp
import pickle
import numpy as np

app=Flask(__name__)
cap=cv2.VideoCapture(0)

coords = Keypoint()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

with open ('Hosting/rf.pkl', 'rb') as f:
    rf = pickle.load(f)

def generate_frames():

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:

                data = coords.extract_keypoints(results)

                X = pd.DataFrame([data])

                X = X.fillna(0)

                class_label = rf.predict(X)[0]

                print(class_label)

                class_probability = rf.predict_proba(X)[0]
                                
                cv2.rectangle(image, (0,0), (100+len(class_label)*20,70), (256, 7, 3), -1)

                cv2.putText(image, 'Prob', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Class', (95,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

                cv2.putText(image,  str(round(class_probability[np.argmax(class_probability)],2)), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)            
                cv2.putText(image, class_label, (95,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:

                #print(e)
                pass

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=False)