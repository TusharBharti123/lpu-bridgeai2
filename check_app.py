from flask import Flask, render_template, Response,request,jsonify
from tensorflow.keras.models import load_model
import cv2
import base64
import numpy as np
import mediapipe as mp
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

auto_correct = genai.GenerativeModel('gemini-1.5-flash')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

model = load_model('model.h5')

app=Flask(__name__)
camera = cv2.VideoCapture(0)
prediction = ''
l=[]
counter = 0

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            predicted_character2 = None
            global prediction
            global l
            global counter

            data_aux1 = []
            x_ = []
            y_ = []
            roi = np.ones((200,200,3))*255
            
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                counter = 0
                for hand_landmarks in results.multi_hand_landmarks[0:1]:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux1.append(x - min(x_))
                        data_aux1.append(y - min(y_))

                x1 = int(min(x_) * W) - 50
                y1 = int(min(y_) * H) - 50

                x2 = int(max(x_) * W) + 50
                y2 = int(max(y_) * H) + 50

                if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
                    roi = frame[y1:y2, x1:x2]
                    roi = cv2.resize(roi, (200, 200))
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    results1 = hands.process(roi)
                    if results1.multi_hand_landmarks:
                        for hand_landmarks in results1.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                roi, 
                                hand_landmarks, 
                                mp_hands.HAND_CONNECTIONS, 
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )

                lal = np.expand_dims(roi, axis=0)
                prediction1 = model.predict(lal)
                data_aux1 = []
                x_ = []
                y_ = []

                predicted_output = np.argmax(prediction1)
                l.append(predicted_output)
                predicted_character1 = chr(predicted_output + ord('A'))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 5)
                cv2.putText(frame, predicted_character1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            else:
                counter += 1
                if counter > 3 and len(l) > 0:
                    predicted_output1 = max(set(l), key=l.count)
                    predicted_character = chr(predicted_output1 + ord('A'))
                    if l.count(predicted_output1) > 2:
                        prediction += predicted_character
                        print(l, prediction)
                        l = []
                        counter = 0

            frame1 = cv2.resize(frame, (400, 400))
            roi1 = cv2.resize(roi, (400, 400))
            combined_image = np.hstack((frame1, np.ones((400, 400, 3)) * 255, roi1))
            ret, buffer1 = cv2.imencode('.jpg', combined_image)
            image_data = base64.b64encode(buffer1).decode('utf-8')

            if len(l) > 0:
                predicted_output2 = max(set(l), key=l.count)
                predicted_character2 = chr(predicted_output2 + ord('A'))
            else:
                predicted_character2 = ''

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image_data.encode() + b'\r\n'
                   b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n' + predicted_character2.encode() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)
    
