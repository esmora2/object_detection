import cv2
import numpy as np
import tensorflow as tf
import pygame



# Inicializar pygame para manejar la alarma
pygame.mixer.init()


# Cargar la alarma
pygame.mixer.music.load('alarm/alarm.mp3')

alarm_playing = False

# Cargar parámetros de calibración de la cámara
with np.load('./calib/calibration_parameters.npz') as data:
    mtx = data['camera_matrix']  # Matriz de la cámara
    dist = data['dist_coeffs']  # Coeficientes de distorsión

# Configuración ArUco
ARUCO_DICT = cv2.aruco.DICT_4X4_250
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT)
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.adaptiveThreshWinSizeMin = 5
arucoParams.adaptiveThreshWinSizeMax = 23
arucoParams.adaptiveThreshWinSizeStep = 10
arucoParams.minMarkerPerimeterRate = 0.03
arucoParams.polygonalApproxAccuracyRate = 0.03
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# Cargar el modelo preentrenado
PATH_TO_SAVED_MODEL = "ssdlite_mobilenet_v2_coco/saved_model"
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
detect_fn = model.signatures['serving_default']

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

def order_points(points):
    points = np.array(points)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = points[np.argmin(s)]
    ordered[2] = points[np.argmax(s)]
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered

def is_point_inside_polygon(polygon, point):
    polygon = np.array(polygon, dtype=np.float32)
    point = np.array(point, dtype=np.float32)
    return cv2.pointPolygonTest(polygon, tuple(point), False) >= 0

last_polygon = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print("Error: No se pudo leer la imagen de la cámara.")
        break

    h, w, _ = img.shape

    # Corregir la distorsión de la imagen
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # Convertir a escala de grises y suavizar la imagen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detección de ArUcos
    corners, ids, _ = cv2.aruco.detectMarkers(blurred, arucoDict, parameters=arucoParams)
    cv2.aruco.drawDetectedMarkers(img, corners, ids)  # Dibujar marcadores detectados para depuración

    if ids is not None and len(ids) >= 4:
        # Ordenar los ArUcos por su ID para mantener un orden consistente
        ids_and_corners = sorted(zip(ids.flatten(), corners), key=lambda x: x[0])
        ordered_ids, ordered_corners = zip(*ids_and_corners[:4])  # Tomar solo los primeros 4 si hay más de 4
        ordered_corners = [corner[0] for corner in ordered_corners]
        flat_corners = [point for sublist in ordered_corners for point in sublist]
        polygon = order_points(flat_corners).astype(int)

        if last_polygon is None or not np.allclose(polygon, last_polygon, atol=5):
            last_polygon = polygon

    # Detección de objetos
    image_np = np.asarray(img)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    num_detections = int(detections['num_detections'].numpy())
    boxes = detections['detection_boxes'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0].astype(int)
    scores = detections['detection_scores'].numpy()[0]

    area_restringida = False

    for i in range(num_detections):
        if scores[i] > 0.5 and classes[i] == 1:  # Persona detectada
            box = boxes[i] * [h, w, h, w]
            ymin, xmin, ymax, xmax = box.astype(int)
            
            # Puntos de las esquinas de la caja delimitadora
            box_points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            
            # Verificar si cualquier punto de la caja delimitadora está dentro del área restringida
            object_in_area = False
            for point in box_points:
                if last_polygon is not None and is_point_inside_polygon(last_polygon, point):
                    object_in_area = True
                    break  # Salir del bucle si al menos un punto está dentro del área

            # Dibujar caja y verificar si está en el área restringida
            if object_in_area:
                area_restringida = True
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            else:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Dibujar el área delimitada por los ArUcos
    if last_polygon is not None:
        cv2.polylines(img, [last_polygon], isClosed=True, color=(0, 0, 255), thickness=2)

    # Manejo de la alarma
    if area_restringida and not alarm_playing:
        pygame.mixer.music.play(-1)  # Reproducir en bucle
        alarm_playing = True
    elif not area_restringida and alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False

    # Mostrar mensaje de área restringida
    if area_restringida:
        cv2.putText(img, "AREA RESTRINGIDA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar imagen
    cv2.imshow("Image", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()