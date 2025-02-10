import cv2
import dlib # its binded from c++ to py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import time
from threading import Thread
from scipy.spatial import distance as dist
import pygame

#NEO-6M GPS خرید
#ماژول های GSM یا LTE:
#SIM800L یا SIM900

EAR_THRESHOLD = 0.24
EAR_SECONDS = 3  # seconds

EMAIL_ADDRESS = 'hexagon.detection.alarm@gmail.com'
EMAIL_PASSWORD = 'xyglcvuiosrsvpsr'
RECEIVER_EMAIL = 'amirhosseinabbasifar@gmail.com'

pygame.mixer.init()

ALARM_SOUND_PATH = "Alarm.mp3"
pygame.mixer.music.load(ALARM_SOUND_PATH)

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def send_email_with_image(image_path):
    def send_email():
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = RECEIVER_EMAIL
            msg['Subject'] = 'Drowsiness Alert!'

            body = 'The system detected prolonged eye closure. Please take a break!'
            msg.attach(MIMEText(body, 'plain'))

            with open(image_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={image_path.split("/")[-1]}',
            )
            msg.attach(part)

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print("Email sent successfully!")
        except Exception as e:
            print(f"Error sending email: {e}")

    Thread(target=send_email).start()

def play_alarm_sound():
    def play():
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    Thread(target=play).start()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(l_start, l_end) = (42, 48)
(r_start, r_end) = (36, 42)

cap = cv2.VideoCapture(0)

closed_eyes_start = None
alarm_triggered = False

while True:
    #strat_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = []
        right_eye = []

        for i in range(l_start, l_end):
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        for i in range(r_start, r_end):
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            if closed_eyes_start is None:
                closed_eyes_start = time.time()
            elif time.time() - closed_eyes_start >= EAR_SECONDS:
                if not alarm_triggered:
                    print("Drowsiness detected! Triggering alarm and sending email...")

                    image_path = "drowsiness_alert.jpg"
                    cv2.imwrite(image_path, frame)
                    send_email_with_image(image_path)

                    play_alarm_sound()

                    alarm_triggered = True
        else:
            closed_eyes_start = None
            alarm_triggered = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    #stop_time = time.time()
    #print(stop_time - strat_time)
    #exit()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
