import cv2
import numpy as np

# Cargar los parámetros de calibración
with np.load('calibration_parameters.npz') as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# Capturar video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: No se pudo capturar el video.")
        break

    # Mostrar el video original (sin calibración)
    cv2.imshow('Original', frame)
    
    # Aplicar la calibración de la cámara
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Recortar la imagen calibrada si es necesario
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]
    
    # Mostrar el video con la calibración aplicada
    cv2.imshow('Calibrado', undistorted_frame)
    
    key = cv2.waitKey(1)
    if key % 256 == 27:
        # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
