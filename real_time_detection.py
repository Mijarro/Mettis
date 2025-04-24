import cv2
import threading
import queue
import time
import os
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime
import csv

# -------------------------------
# CONFIGURACIÓN Y PREPARACIÓN
# -------------------------------

# Tamaño de los fotogramas a procesar
FRAME_SIZE = (640, 640)

# Nombre de la ventana para mostrar el vídeo
WINDOW_NAME = "Detección y Segmentación en Tiempo Real"

# FPS deseado para el procesamiento del vídeo
FPS = 20

# Generar un timestamp único para cada ejecución
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Definir las rutas de salida para el vídeo y el log CSV
OUTPUT_PATH = f"video_salida_{timestamp}.mp4"
CSV_LOG_PATH = f"rendimiento_{timestamp}.csv"

# Crear las carpetas necesarias para almacenar los resultados de baja confianza
ROOT_RETRAIN_DIR = f"dataset_reentrenamiento_{timestamp}"
DET_IMG_DIR = os.path.join(ROOT_RETRAIN_DIR, "images_det")
DET_LBL_DIR = os.path.join(ROOT_RETRAIN_DIR, "labels_det")
SEG_IMG_DIR = os.path.join(ROOT_RETRAIN_DIR, "images_seg")
SEG_LBL_DIR = os.path.join(ROOT_RETRAIN_DIR, "labels_seg")

# Crear las carpetas si no existen
for path in [DET_IMG_DIR, DET_LBL_DIR, SEG_IMG_DIR, SEG_LBL_DIR]:
    os.makedirs(path, exist_ok=True)

# Definir las rutas de los modelos
DETECTION_MODEL_PATH = "best_det.pt"
ROAD_MODEL_PATH = "best_seg.pt"

# Verificar si los modelos existen
if not os.path.exists(DETECTION_MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo de detección: {DETECTION_MODEL_PATH}")
if not os.path.exists(ROAD_MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo de carretera: {ROAD_MODEL_PATH}")

# Establecer el dispositivo (GPU si está disponible, sino CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cargar los modelos de detección y segmentación
detection_model = YOLO(DETECTION_MODEL_PATH).to(device)
road_model = YOLO(ROAD_MODEL_PATH).to(device)

# Definir los colores para las anotaciones de detección y segmentación
COLORS = {
    'detection': (0, 255, 0),  # Verde para detecciones
    'drivable area': (0, 0, 255),  # Rojo para áreas transitable
    'lane': (255, 0, 0),  # Azul para las líneas
    'text': (255, 255, 255)  # Blanco para el texto
}

# -------------------------------
# FUNCIONES DE PROCESAMIENTO Y DIBUJADO
# -------------------------------

def process_frame(model, frame, is_segmentation=False):
    """
    Procesa un fotograma con el modelo de detección o segmentación.
    Devuelve las predicciones de la detección o segmentación.
    """
    # Realizar la predicción con el modelo
    results = model(frame, verbose=False, imgsz=640, conf=0.25, iou=0.45,
                   device='0' if torch.cuda.is_available() else 'cpu', half=True)
    result = results[0]
    predictions = []
    
    # Si es segmentación, procesar las máscaras
    if is_segmentation and hasattr(result, 'masks') and result.masks is not None:
        for mask, box in zip(result.masks.xy, result.boxes):
            predictions.append({
                'class': model.names[int(box.cls)],  # Nombre de la clase
                'segmentation': mask,  # Máscara de segmentación
                'confidence': float(box.conf),  # Confianza
                'class_id': int(box.cls)  # ID de clase
            })
    else:
        # Si es detección, procesar las coordenadas de los cuadros
        for box in result.boxes:
            predictions.append({
                'class': model.names[int(box.cls)],  # Nombre de la clase
                'coordinates': [int(coord) for coord in box.xyxy[0].tolist()],  # Coordenadas del cuadro
                'confidence': float(box.conf),  # Confianza
                'class_id': int(box.cls)  # ID de clase
            })
    
    return {'predictions': predictions}

def draw_detections(frame, detections):
    """
    Dibuja los cuadros de detección sobre el fotograma.
    """
    for pred in detections['predictions']:
        if 'coordinates' not in pred:
            continue
        # Obtener las coordenadas del cuadro
        x1, y1, x2, y2 = pred['coordinates']
        # Dibujar el cuadro sobre el fotograma
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['detection'], 2)
        # Escribir el nombre de la clase y la confianza sobre el cuadro
        label = f"{pred['class']} {pred['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
    return frame

def draw_segmentations(frame, segmentation_data, alpha=0.4):
    """
    Dibuja las segmentaciones sobre el fotograma.
    """
    overlay = frame.copy()
    for pred in segmentation_data.get('predictions', []):
        mask = pred.get('segmentation')
        class_name = pred.get('class', 'unknown')
        confidence = pred.get('confidence', 0.0)
        if mask is None or len(mask) == 0:
            continue
        color = COLORS.get(class_name, (128, 128, 128))  # Color de la clase
        pts = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))  # Coordenadas de la máscara
        cv2.fillPoly(overlay, [pts], color)  # Rellenar la máscara
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)  # Dibujar el contorno
        x, y = pts[0][0]
        label = f"{class_name} {confidence:.2f}"  # Etiqueta con la clase y confianza
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Superponer la máscara con el fotograma original
    return frame

def guardar_deteccion_baja_confianza(frame_id, frame, detections):
    """
    Guarda las detecciones con baja confianza en un archivo.
    """
    img_path = os.path.join(DET_IMG_DIR, f"frame_{frame_id:06d}.jpg")
    lbl_path = os.path.join(DET_LBL_DIR, f"frame_{frame_id:06d}.txt")
    cv2.imwrite(img_path, frame)  # Guardar la imagen
    with open(lbl_path, 'w') as f:
        for pred in detections['predictions']:
            if 'coordinates' not in pred or pred['confidence'] >= 0.6:  # Baja confianza
                continue
            x1, y1, x2, y2 = pred['coordinates']
            cx = (x1 + x2) / 2 / FRAME_SIZE[0]  # Coordenada x normalizada
            cy = (y1 + y2) / 2 / FRAME_SIZE[1]  # Coordenada y normalizada
            w = (x2 - x1) / FRAME_SIZE[0]  # Ancho normalizado
            h = (y2 - y1) / FRAME_SIZE[1]  # Alto normalizado
            f.write(f"{pred['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")  # Escribir la etiqueta en formato YOLO

def guardar_segmentacion_baja_confianza(frame_id, frame, segmentations):
    """
    Guarda las segmentaciones con baja confianza en un archivo.
    """
    img_path = os.path.join(SEG_IMG_DIR, f"frame_{frame_id:06d}.jpg")
    lbl_path = os.path.join(SEG_LBL_DIR, f"frame_{frame_id:06d}.txt")
    cv2.imwrite(img_path, frame)  # Guardar la imagen
    with open(lbl_path, 'w') as f:
        for pred in segmentations['predictions']:
            if 'segmentation' not in pred or pred['confidence'] >= 0.5:  # Baja confianza
                continue
            class_id = pred['class_id']
            pts = np.array(pred['segmentation'], dtype=np.float32)  # Coordenadas de la segmentación
            flat_pts = pts.flatten()  # Aplanar las coordenadas
            coords_str = ' '.join([f"{x:.2f}" for x in flat_pts])  # Convertir a string
            f.write(f"{class_id} {coords_str}\n")  # Escribir la segmentación

def main():
    """
    Función principal que gestiona el procesamiento de vídeo en tiempo real,
    la detección y segmentación de objetos, y el guardado de resultados.
    """
    cap = cv2.VideoCapture(1)  # Captura de vídeo desde la cámara
    cap.set(cv2.CAP_PROP_FPS, FPS)  # Configuración del FPS
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # Crear ventana para mostrar el vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codificación de vídeo
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, FRAME_SIZE)  # Salida de vídeo

    # Abrir el archivo CSV para registrar el rendimiento
    with open(CSV_LOG_PATH, mode='w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(["frame_index", "start_time", "end_time", "latencia_segundos"])  # Cabecera del log

        frame_index = 0  # Índice de los fotogramas procesados
        try:
            while True:
                start = time.time()  # Tiempo de inicio del procesamiento de cada fotograma
                ret, frame = cap.read()
                if not ret:
                    print("No se pudo leer el frame de la cámara.")
                    break

                frame_resized = cv2.resize(frame, FRAME_SIZE)  # Redimensionar el fotograma
                detections = process_frame(detection_model, frame_resized)  # Procesar detección
                segmentations = process_frame(road_model, frame_resized, is_segmentation=True)  # Procesar segmentación

                # Dibujar detecciones y segmentaciones en el fotograma
                frame_annotated = draw_detections(frame_resized.copy(), detections)
                frame_annotated = draw_segmentations(frame_annotated, segmentations)

                # Guardar las detecciones o segmentaciones con baja confianza
                if any(p.get('confidence', 1) < 0.6 for p in detections['predictions'] if 'coordinates' in p):
                    guardar_deteccion_baja_confianza(frame_index, frame_resized, detections)
                if any(p.get('confidence', 1) < 0.5 for p in segmentations['predictions'] if 'segmentation' in p):
                    guardar_segmentacion_baja_confianza(frame_index, frame_resized, segmentations)

                out.write(frame_annotated)  # Escribir el fotograma anotado en el archivo de salida
                cv2.imshow(WINDOW_NAME, frame_annotated)  # Mostrar el fotograma en la ventana

                end = time.time()  # Tiempo de fin del procesamiento
                latency = end - start  # Calcular latencia
                log_writer.writerow([frame_index, start, end, latency])  # Registrar el rendimiento
                frame_index += 1  # Incrementar el índice del fotograma

                # Calcular el tiempo de espera para mantener el FPS deseado
                wait = max(1, int((1 / FPS - latency) * 1000))
                if cv2.waitKey(wait) & 0xFF == ord('q'):  # Salir si se presiona la tecla 'q'
                    break
        finally:
            cap.release()  # Liberar la cámara
            out.release()  # Liberar el archivo de vídeo
            cv2.destroyAllWindows()  # Cerrar las ventanas
            print(f"Vídeo guardado en: {OUTPUT_PATH}")
            print(f"Log de rendimiento guardado en: {CSV_LOG_PATH}")
            print(f"Frames con baja confianza guardados en: {ROOT_RETRAIN_DIR}")

if __name__ == "__main__":
    main()  # Ejecutar la función principal
