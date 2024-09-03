import cv2
import os
import time

# Verificar si el directorio de almacenamiento existe, si no, crearlo
output_dir = 'calibration_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iniciar la captura de video
cap = cv2.VideoCapture(0)
img_counter = 0

if not cap.isOpened():
    print("Error: No se puede acceder a la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo obtener la imagen de la cámara.")
        break

    cv2.imshow('Captura de Calibración', frame)

    key = cv2.waitKey(1)
    
    if key % 256 == 27:
        # Presiona ESC para salir
        print("Salida del programa.")
        break
    elif key % 256 == 32:
        # Presiona SPACE para capturar
        img_name = f'{output_dir}/calib_{img_counter}.png'
        cv2.imwrite(img_name, frame)
        print(f'{img_name} guardada!')
        img_counter += 1
        
        # Esperar un segundo antes de permitir otra captura para evitar problemas de almacenamiento rápido
        time.sleep(1)

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
