import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime

# Inicialización
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Para guardar datos
data = []

# Captura de cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Coordenadas de los ojos (ejemplo: ojo izquierdo)
            left_eye_x = face_landmarks.landmark[33].x
            left_eye_y = face_landmarks.landmark[33].y
            timestamp = datetime.now().strftime("%H:%M:%S.%f")
            data.append([timestamp, left_eye_x, left_eye_y])

            h, w, _ = frame.shape
            cx, cy = int(left_eye_x * w), int(left_eye_y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Guardar en CSV
df = pd.DataFrame(data, columns=["timestamp", "x", "y"])
df.to_csv("eye_tracking_data.csv", index=False)

cap.release()
cv2.destroyAllWindows()
