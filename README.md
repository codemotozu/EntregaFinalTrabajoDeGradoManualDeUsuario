# WasteCareAI - Sistema de Clasificación de Residuos mediante Visión Artificial

Sistema de clasificación automática de residuos que utiliza visión artificial y aprendizaje profundo basado en YOLOv11 con segmentación de instancias para identificar 10 categorías diferentes de materiales reciclables en tiempo real.

## 🎥 Demostración
**Video en funcionamiento:** https://www.youtube.com/watch?v=uTHRAp4GBTs

**Modelo entrenado para descargar:** https://huggingface.co/CAROCH/MODEL/tree/main

## 📊 Resultados del Sistema
- **Precisión Global:** mAP@50 = 93.2%
- **Velocidad:** 15-38 FPS (dependiendo del hardware)
- **Latencia:** 28-35ms por frame
- **Categorías:** 10 tipos de residuos detectados

## 🗂️ Categorías de Residuos Detectadas
1. **BotellaDePlastico** - Botellas PET transparentes
2. **ContenedorDePlastico** - Envases HDPE opacos
3. **BolsaDePlastico** - Bolsas y plásticos flexibles
4. **Papel** - Material celulósico de bajo gramaje
5. **Cartón** - Material celulósico de alto gramaje
6. **Metal** - Latas de aluminio y envases metálicos
7. **VidrioTransparente** - Envases de vidrio incoloro
8. **VidrioMarron** - Envases de vidrio ámbar
9. **VidrioVerde** - Envases de vidrio verde
10. **CepilloDeDientes** - Artículos de higiene personal

## 🛠️ Requisitos del Sistema

### Hardware Mínimo
- **GPU:** NVIDIA con 8GB VRAM y soporte CUDA 11.4+
- **CPU:** 6 núcleos/12 hilos a 3.5GHz o superior
- **RAM:** 16GB DDR4-3200 o superior
- **Almacenamiento:** SSD NVMe recomendado
- **Cámara:** Webcam o cámara USB

### Software
- **Python:** 3.8 o superior
- **CUDA Toolkit:** 11.4 o superior (para GPU)
- **Sistema Operativo:** Windows 10/11, Linux, macOS

## 🚀 Instalación

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/codemotozu/EntregaFinalTrabajoDeGradoManualDeUsuario.git
cd EntregaFinalTrabajoDeGradoManualDeUsuario
```
### Paso 2: Crear Entorno Virtual
```bash
# Crear entorno virtual
python -m venv wastecareai_env

# Activar entorno virtual
# Windows:
wastecareai_env\Scripts\activate
# Linux/macOS:
source wastecareai_env/bin/activate
```

### Paso 3: Instalar Dependencias
```bash
# Instalar PyTorch con soporte CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar otras dependencias
pip install ultralytics
pip install opencv-python
pip install numpy
```

### Paso 4: Descargar Modelo Entrenado
1. Ir a: https://huggingface.co/CAROCH/MODEL/tree/main
2. Descargar `bestModel09052025.pt`
3. Colocar el archivo en una carpeta accesible

### Paso 5: Configurar Ruta del Modelo
Editar la línea 26 en `yolov11.py`:
```python
model = YOLO(r"RUTA_COMPLETA_AL_MODELO/bestModel09052025.pt")
```

## 🎮 Uso del Sistema

### Ejecutar el Sistema
```bash
python yolov11.py
```

### Controles
- **Iniciar:** Ejecutar el script automáticamente inicia la detección
- **Salir:** Presionar `'q'` en la ventana de video

### Métricas en Pantalla
- **FPS Visualización:** Cuadros mostrados por segundo (25-60 FPS típico)
- **FPS Procesamiento:** Velocidad real de análisis (15-38 FPS según hardware)
- **Inferencia:** Tiempo de procesamiento por frame (28-35ms)
- **Dispositivo:** GPU o CPU en uso

## ⚙️ Configuración Avanzada

### Ajustar Parámetros de Detección
En `yolov11.py`, líneas 35-39:
```python
results = model.predict(
    source=frame_buffer.copy(),
    conf=0.5,      # Umbral de confianza (0.3-0.8 recomendado)
    iou=0.45,      # Umbral IoU (0.4-0.6 recomendado)
    max_det=10,    # Máximo objetos detectados (5-15)
    verbose=False
)
```

### Cambiar Cámara
Si tienes múltiples cámaras, cambiar el índice en línea 60:
```python
cap = cv2.VideoCapture(0)  # 0, 1, 2, etc.
```

## 🔧 Solución de Problemas

### Problemas Comunes

**"GPU no disponible, usando CPU"**
- Verificar instalación de CUDA Toolkit
- Actualizar drivers NVIDIA
- Comprobar compatibilidad de GPU

**FPS bajo (<10 FPS)**
- Reducir resolución de cámara
- Ajustar parámetros `max_det` y `conf`
- Cerrar aplicaciones que usen GPU

**No detecta la cámara**
- Verificar conexión USB
- Cambiar índice de cámara (0, 1, 2...)
- Cerrar otras aplicaciones que usen la cámara

**Detecciones incorrectas**
- Mejorar iluminación
- Limpiar lente de cámara
- Ajustar umbral de confianza

## 📁 Estructura del Proyecto
```
EntregaFinalTrabajoDeGradoManualDeUsuario/
├── yolov11.py                 # Código principal del sistema
├── README.md                  # Este archivo
├── bestModel09052025.pt       # Modelo entrenado (descargar por separado)
└── requirements.txt           # Dependencias (opcional)
```

## 📊 Rendimiento por Hardware
| Hardware | Resolución | FPS Procesamiento | Latencia |
|----------|------------|-------------------|----------|
| RTX 4080 | 640×640 | 38.7 FPS | 25.8ms |
| RTX 3060 | 640×640 | 26.3 FPS | 38.0ms |
| Jetson AGX Orin | 640×640 | 22.1 FPS | 45.2ms |
| RTX T4 (Cloud) | 640×640 | 24.6 FPS | 40.7ms |

## 🤝 Contribución
Este es el trabajo de grado final del proyecto WasteCareAI. Para consultas técnicas o colaboraciones, contactar a través del repositorio.

## 📄 Licencia
Este proyecto fue desarrollado como trabajo de grado académico en el Politécnico Grancolombiano.

## 🔗 Enlaces Relacionados
- **Video Demostración:** https://www.youtube.com/watch?v=uTHRAp4GBTs
- **Modelo Entrenado:** https://huggingface.co/CAROCH/MODEL/tree/main
- **Documentación YOLOv11:** https://docs.ultralytics.com/


