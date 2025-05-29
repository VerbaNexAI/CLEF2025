# CLEF Touché 2025 – Image Retrieval for Arguments

![CEDNAV Logo](https://github.com/HIPERDAGA/EO-Target-Tracking/blob/main/LOGO_CEDNAV_02.gif?raw=true)
![Firma](https://github.com/HIPERDAGA/EO-Target-Tracking/blob/main/firma_animada.gif?raw=true)

Este repositorio contiene el código para la participación del equipo **Computer Vision UTB**, afiliado al **Centro de Desarrollo Tecnológico Naval (CEDNAV)**, en la tarea de recuperación de imágenes para argumentos del desafío [Touché 2025](https://touche.webis.de/clef25/touche25-web/image-retrieval-for-arguments.html).

---

## 📌 Descripción

El sistema implementa una arquitectura de recuperación de imágenes basada en similitud semántica usando [CLIP (ViT-B/32)](https://openai.com/research/clip). Dado un conjunto de argumentos (claims) y un conjunto de imágenes con captions, el sistema:

1. Embebe los textos de los claims y los captions con CLIP.
2. Calcula la similitud coseno entre embeddings.
3. Recupera las 10 imágenes más relevantes por claim.
4. Genera un archivo `submission.jsonl` con las predicciones, conforme al formato del reto.

Además, se incluye trazabilidad de huella de carbono mediante [CodeCarbon](https://mlco2.github.io/codecarbon/).

---

## 📁 Estructura del proyecto

├── Touché2025_V7.ipynb # Notebook principal con el pipeline completo
├── /DATASET_TOUCHE_2025 # Carpeta en Google Drive con los datos y embeddings
│ ├── touche25-image-retrieval-and-generation-main.zip
│ ├── arguments.xml
│ ├── claim_embeddings.pt
│ ├── caption_embeddings.pt
│ └── submission.jsonl


---

## ⚙️ Requisitos

- Google Colab
- Python 3.8+
- PyTorch
- `open_clip_torch`
- `codecarbon`
- `tqdm`

Instalación en Colab:

```bash
!pip install open_clip_torch codecarbon tqdm


🚀 Ejecución
Montar Google Drive y definir rutas

Descomprimir el dataset si no se ha hecho previamente

Cargar argumentos (arguments.xml)

Cargar captions de las imágenes usando multiprocesamiento

Configurar modelo CLIP (ViT-B/32)

Generar o cargar embeddings de claims y captions

Calcular similitud coseno y recuperar imágenes top-10

Guardar archivo de resultados en formato JSONL

Medir emisiones de carbono con CodeCarbon

📤 Formato de salida
Cada línea del archivo submission.jsonl contiene una predicción:


{
  "argument_id": "001",
  "method": "retrieval",
  "image_id": "I1234",
  "rank": 1,
  "tag": "CEDNAV-UTB; CLIP_Baseline"
}

📊 Métricas
La evaluación se realiza mediante nDCG@10 sobre la correspondencia entre imágenes recuperadas y relevancia dada por anotadores humanos

🌱 Huella de carbono
El pipeline registra el impacto ambiental estimado (emisiones de CO₂ en kg) con codecarbon.

👥 Equipo
Nombre del equipo: Computer Vision UTB

Afiliación: CEDNAV

País: Colombia

Contacto: Diego Guevara

Email: hiperdaga7@gmail.com


📝 Licencia
Este proyecto se distribuye con fines académicos. Revisa las condiciones de uso del dataset Touché 2025 antes de su reutilización.

yaml
Copiar
Editar

