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


