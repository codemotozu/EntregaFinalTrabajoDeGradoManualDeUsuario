import time  # Importa módulo para medir tiempos y pausas
import torch # Importa PyTorch para operaciones de deep learning y manejo de GPU
import cv2 # Importa OpenCV para captura y procesamiento de video
import threading # Importa threading para ejecutar procesamiento en paralelo
import numpy as np # Importa NumPy para operaciones con arrays numéricos
from ultralytics import YOLO # Importa la clase YOLO de Ultralytics para detección de objetos

# Verificar disponibilidad de CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Detecta si hay GPU disponible, si no usa CPU
if device == 'cuda': # Si hay GPU disponible
    torch.cuda.set_device(0) # Selecciona la primera GPU (índice 0)
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}") # Muestra el nombre de la GPU
else: # Si no hay GPU
    print("GPU no disponible, usando CPU") # Informa que usará CPU

# Variables globales para comunicación entre hilos
latest_results = None # Almacena los últimos resultados de detección del modelo
processing = False # Bandera que indica si el modelo está procesando un frame
fps_display = 0 # Almacena los FPS de procesamiento para mostrar en pantalla
inference_time_ms = 0 # Tiempo que tarda el modelo en procesar un frame (en milisegundos)
frame_buffer = None # Buffer que almacena el frame más reciente para procesar
running = True # Bandera que controla si el sistema sigue ejecutándose

# Función para procesar frames en segundo plano
def process_frames(): # Define función que se ejecutará en un hilo separado
    global latest_results, processing, fps_display, inference_time_ms, running # Accede a variables globales
    
    model = YOLO(r"C:\Users\rodri\Downloads\bestModel09052025.pt") # ⚠️ IMPORTANTE: Copy-paste tu path donde descargaste el modelo que se encuentra en el readme 🔄 CAMBIAR ESTA RUTA
    model.to(device) # Mueve el modelo a GPU o CPU según disponibilidad
    
    fps_list = [] # Lista para calcular promedio de FPS
    prev_time = time.time() # Guarda tiempo anterior para calcular FPS
    
    while running: # Bucle principal del hilo de procesamiento
        if frame_buffer is not None and not processing: # Si hay un frame disponible y no se está procesando
            processing = True # Marca que está procesando
            
            # Iniciar cronómetro para inferencia
            inference_start = time.time() # Guarda tiempo de inicio de inferencia
            
            # Procesar el frame con YOLO
            results = model.predict( # Ejecuta predicción del modelo YOLO
                source=frame_buffer.copy(), # Frame de entrada (copia para evitar modificaciones)
                conf=0.5, # Umbral de confianza mínimo (50%)
                iou=0.45, # Umbral de IoU para supresión de no-máximos
                max_det=10, # Máximo número de detecciones por frame
                verbose=False # No mostrar información detallada en consola
            )
            
            # Guardar resultados para que el hilo principal los muestre
            latest_results = results[0] # Guarda el primer resultado (hay solo uno por frame)
            
            # Calcular tiempo de inferencia
            inference_time_ms = (time.time() - inference_start) * 1000 # Calcula tiempo en milisegundos
            
            # Calcular FPS de procesamiento
            current_time = time.time() # Tiempo actual
            fps = 1 / (current_time - prev_time) # Calcula FPS instantáneo
            fps_list.append(fps) # Añade a la lista
            if len(fps_list) > 10: # Si la lista tiene más de 10 elementos
                fps_list.pop(0) # Elimina el más antiguo
            fps_display = sum(fps_list) / len(fps_list) # Calcula promedio de FPS
            
            prev_time = current_time # Actualiza tiempo anterior
            processing = False # Marca que terminó de procesar
            
            # Breve pausa para evitar sobrecarga de CPU/GPU
            time.sleep(0.01) # Pausa de 10ms para no saturar el sistema

# Iniciar hilo de procesamiento
process_thread = threading.Thread(target=process_frames) # Crea hilo para procesar frames
process_thread.daemon = True # Marca como hilo daemon (se cierra al terminar programa principal)
process_thread.start() # Inicia el hilo de procesamiento

# Captura de video en hilo principal
cap = cv2.VideoCapture(0) # Inicia captura de video desde cámara (índice 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Establece ancho de captura en 640 píxeles
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Establece alto de captura en 480 píxeles

# Variables para FPS de visualización
display_fps = 0 # FPS de visualización (cuántos frames se muestran por segundo)
display_prev_time = time.time() # Tiempo anterior para calcular FPS de visualización
display_fps_list = [] # Lista para promedio de FPS de visualización

try: # Inicia bloque try para manejo de errores
    while cap.isOpened(): # Mientras la cámara esté disponible
        ret, frame = cap.read() # Captura un frame de la cámara
        if not ret: # Si no se pudo capturar el frame
            break # Sale del bucle
        
        # Actualizar buffer de frame para el hilo de procesamiento
        frame_buffer = frame.copy() # Copia el frame actual al buffer global
        
        # Preparar frame para mostrar (con o sin anotaciones)
        display_frame = frame.copy() # Crea copia del frame para mostrar
        
        # Aplicar anotaciones de la última detección a TODOS los frames
        if latest_results is not None: # Si hay resultados de detección disponibles
            display_frame = latest_results.plot(img=display_frame) # Dibuja cajas y etiquetas en el frame
        
        # Calcular FPS de visualización (siempre será mayor que FPS de procesamiento)
        current_time = time.time() # Tiempo actual
        display_frame_time = current_time - display_prev_time # Tiempo desde último frame mostrado
        display_prev_time = current_time # Actualiza tiempo anterior
        
        if display_frame_time > 0: # Si el tiempo es válido
            display_fps_list.append(1 / display_frame_time) # Calcula y añade FPS instantáneo
            if len(display_fps_list) > 30: # Si la lista tiene más de 30 elementos
                display_fps_list.pop(0) # Elimina el más antiguo
            display_fps = sum(display_fps_list) / len(display_fps_list) # Calcula promedio
        
        # Agregar información en pantalla (constante en todos los frames)
        cv2.putText(display_frame, f"FPS Visualizacion: {int(display_fps)}", (20, 30), # Dibuja FPS de visualización
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Fuente, tamaño, color verde, grosor
        cv2.putText(display_frame, f"FPS Procesamiento: {int(fps_display)}", (20, 60), # Dibuja FPS de procesamiento
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Mismos parámetros de texto
        cv2.putText(display_frame, f"Inferencia: {inference_time_ms:.1f}ms", (20, 90), # Dibuja tiempo de inferencia
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Con un decimal
        cv2.putText(display_frame, f"Dispositivo: {device.upper()}", (20, 120), # Dibuja dispositivo usado (GPU/CPU)
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # En mayúsculas
        
        # Mostrar frame
        cv2.imshow("YOLOv8 Detección (Optimizado)", display_frame) # Muestra ventana con frame procesado
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'): # Si se presiona la tecla 'q'
            break # Sale del bucle principal

except Exception as e: # Si ocurre algún error
    print(f"Error: {e}") # Muestra el error

finally: # Bloque que siempre se ejecuta al final
    # Detener hilo de procesamiento
    running = False # Cambia bandera para detener hilo de procesamiento
    if process_thread.is_alive(): # Si el hilo sigue ejecutándose
        process_thread.join(timeout=1.0) # Espera máximo 1 segundo a que termine
    
    # Liberar recursos
    cap.release() # Libera la cámara
    cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV
    print(f"FPS Visualización: {display_fps:.2f}, FPS Procesamiento: {fps_display:.2f}") # Muestra estadísticas finales
