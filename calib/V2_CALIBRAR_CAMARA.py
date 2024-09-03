import cv2
import numpy as np
import glob

# Definir el tamaño del tablero y el tamaño de los cuadrados
chessboard_size = (9, 7)  # Número de esquinas internas
square_size = 20  # Tamaño de cada cuadrado en mm

# Criterios de terminación
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparar puntos de objeto
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays para almacenar puntos de objeto y puntos de imagen de todas las imágenes
objpoints = []  # Puntos 3D en el mundo real
imgpoints = []  # Puntos 2D en el plano de la imagen

# Cargar imágenes
images = glob.glob('./calibration_images/*.png')

if not images:
    print("Error: No se encontraron imágenes en el directorio especificado.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Encontrar las esquinas del tablero
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        print(f"Patrón encontrado en: {fname}")
        objpoints.append(objp)
        
        # Refinar la ubicación de las esquinas
        corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners_subpix)
        
        # Dibujar y mostrar las esquinas
        cv2.drawChessboardCorners(img, chessboard_size, corners_subpix, ret)
        cv2.imshow('Esquinas detectadas', img)
        cv2.waitKey(500)
    else:
        print(f"No se pudo encontrar el patrón en: {fname}")

cv2.destroyAllWindows()

if not objpoints or not imgpoints:
    print("Error: No se encontraron suficientes puntos para la calibración. Verifica tus imágenes y el tamaño del tablero.")
    exit()

# Calibrar la cámara
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Mostrar resultados
print("Matriz de la cámara:")
print(camera_matrix)
print("\nCoeficientes de distorsión:")
print(dist_coeffs)

# Guardar los parámetros de calibración
np.savez('calibration_parameters.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
