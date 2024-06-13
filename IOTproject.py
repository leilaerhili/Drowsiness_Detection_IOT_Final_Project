from flask import Flask, Response, render_template_string
import cv2
import dlib
from imutils import face_utils
import numpy as np
from scipy.spatial import distance as dist
import threading
import os

app = Flask(__name__)

#Initialize dlib's face detector and load the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')#Update the path to your predictor

#Define the indices for the left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def generate_frames():
    camera = cv2.VideoCapture(0)
    total = 0
    alarm = False

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < 0.25:
                    total += 1
                    if total >= 20:
                        if not alarm:
                            alarm = True
                            threading.Thread(target=lambda: os.system('play alarm.wav')).start()#Update this path or method to play sound
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    total = 0
                    alarm = False

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  #Concatenate frame data

    camera.release()

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    #Return a simple HTML template that includes the video stream
    return render_template_string("""
    <html>
    <head>
    <title>Live Video Stream with Drowsiness Detection</title>
    </head>
    <body>
    <h1>Live Stream from Webcam with Drowsiness Detection</h1>
    <img src="/video" width="640" height="480" />
    </body>
    </html>
    """)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
