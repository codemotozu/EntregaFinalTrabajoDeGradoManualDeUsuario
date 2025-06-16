import time
import torch
import cv2
import threading
import numpy as np
from ultralytics import YOLO

# Verificar disponibilidad de CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(0)
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU no disponible, usando CPU")

# Variables globales para comunicación entre hilos
latest_results = None
processing = False
fps_display = 0
inference_time_ms = 0
frame_buffer = None
running = True

# Función para procesar frames en segundo plano
def process_frames():
    global latest_results, processing, fps_display, inference_time_ms, running
    
    model = YOLO(r"C:\Users\rodri\Downloads\bestModel09052025.pt")  # copy paste tu path donde descargaste el modelo que se encuentra en el readme 
    model.to(device)
    
    fps_list = []
    prev_time = time.time()
    
    while running:
        if frame_buffer is not None and not processing:
            processing = True
            
            # Iniciar cronómetro para inferencia
            inference_start = time.time()
            
            # Procesar el frame con YOLO
            results = model.predict(
                source=frame_buffer.copy(),
                conf=0.5,
                iou=0.45,
                max_det=10,
                verbose=False
            )
            
            # Guardar resultados para que el hilo principal los muestre
            latest_results = results[0]
            
            # Calcular tiempo de inferencia
            inference_time_ms = (time.time() - inference_start) * 1000
            
            # Calcular FPS de procesamiento
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            fps_list.append(fps)
            if len(fps_list) > 10:
                fps_list.pop(0)
            fps_display = sum(fps_list) / len(fps_list)
            
            prev_time = current_time
            processing = False
            
            # Breve pausa para evitar sobrecarga de CPU/GPU
            time.sleep(0.01)

# Iniciar hilo de procesamiento
process_thread = threading.Thread(target=process_frames)
process_thread.daemon = True
process_thread.start()

# Captura de video en hilo principal
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables para FPS de visualización
display_fps = 0
display_prev_time = time.time()
display_fps_list = []

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Actualizar buffer de frame para el hilo de procesamiento
        frame_buffer = frame.copy()
        
        # Preparar frame para mostrar (con o sin anotaciones)
        display_frame = frame.copy()
        
        # Aplicar anotaciones de la última detección a TODOS los frames
        if latest_results is not None:
            display_frame = latest_results.plot(img=display_frame)
        
        # Calcular FPS de visualización (siempre será mayor que FPS de procesamiento)
        current_time = time.time()
        display_frame_time = current_time - display_prev_time
        display_prev_time = current_time
        
        if display_frame_time > 0:
            display_fps_list.append(1 / display_frame_time)
            if len(display_fps_list) > 30:
                display_fps_list.pop(0)
            display_fps = sum(display_fps_list) / len(display_fps_list)
        
        # Agregar información en pantalla (constante en todos los frames)
        cv2.putText(display_frame, f"FPS Visualizacion: {int(display_fps)}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"FPS Procesamiento: {int(fps_display)}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Inferencia: {inference_time_ms:.1f}ms", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Dispositivo: {device.upper()}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow("YOLOv8 Detección (Optimizado)", display_frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # Detener hilo de procesamiento
    running = False
    if process_thread.is_alive():
        process_thread.join(timeout=1.0)
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print(f"FPS Visualización: {display_fps:.2f}, FPS Procesamiento: {fps_display:.2f}")


