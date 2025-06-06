import cv2
import mediapipe as mp
import pandas as pd
import time

# Cargar imángenes
ruta_mockup_bueno = 'mockup_bueno.png'
ruta_mockup_malo = 'mockup_malo.png'

img_bueno = cv2.imread(ruta_mockup_bueno)
img_malo = cv2.imread(ruta_mockup_malo)

# Redimensionar imágenes a la mitad de la pantalla 
img_bueno = cv2.resize(img_bueno, (640, 480))
img_malo = cv2.resize(img_malo, (640, 480))

mockup_frame = cv2.hconcat([img_bueno, img_malo])
frame_height, frame_width = mockup_frame.shape[:2]

# Coordenadas de regiones
mockup_bueno_region = (0, 0, 640, frame_height)
mockup_malo_region = (640, 0, 1280, frame_height)

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

mp_drawing = mp.solutions.drawing_utils

# Coordenadas de las pupilas
LEFT_EYE_IDX = 468
RIGHT_EYE_IDX = 473

# Webcam
cap = cv2.VideoCapture(0)

data = []

print("Presiona 'x' para salir...")

def en_region(x, y, region):
    x1, y1, x2, y2 = region
    return x1 <= x <= x2 and y1 <= y <= y2

while True:
    success, frame = cap.read()
    if not success:
        break

    # Redimensionar la cámara al tamaño de los mockups (640x480)
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Convertir a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    timestamp = time.time()
    foco = 'ninguno'

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = face_landmarks.landmark[LEFT_EYE_IDX]
            right_eye = face_landmarks.landmark[RIGHT_EYE_IDX]

            left_x, left_y = int(left_eye.x * w), int(left_eye.y * h)
            right_x, right_y = int(right_eye.x * w), int(right_eye.y * h)

            # Dibujar en la imagen
            cv2.circle(frame, (left_x, left_y), 3, (0, 255, 0), -1)
            cv2.circle(frame, (right_x, right_y), 3, (255, 0, 0), -1)

            # Determinar zona
            if en_region(left_x, left_y, mockup_bueno_region) or en_region(right_x, right_y, mockup_bueno_region):
                foco = 'bueno'
            elif en_region(left_x, left_y, mockup_malo_region) or en_region(right_x, right_y, mockup_malo_region):
                foco = 'malo'

            # Guardar datos
            data.append({
                "timestamp": timestamp,
                "left_eye_x": left_x,
                "left_eye_y": left_y,
                "right_eye_x": right_x,
                "right_eye_y": right_y,
                "zona": foco
            })

    display = cv2.vconcat([mockup_frame, frame])


    cv2.imshow("Eye Tracking con Mockups", display)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Guardar CSV
df = pd.DataFrame(data)
df.to_csv("eye_tracking_mockups.csv", index=False)

# Liberar
cap.release()
cv2.destroyAllWindows()
