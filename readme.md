TumorDetectRN | By: Joel Sigüenza
================================================================

Explicación
-----------
Este proyecto combina técnicas de deep learning con herramientas de desarrollo web para ofrecer una solución integral en la predicción de tipos de tumores.
Incluye:

Un modelo de red neuronal (CNN) entrenado con imágenes de resonancia magnética para identificar y clasificar diferentes tipos de tumores.
Una API desarrollada con FastAPI, que expone el modelo para su integración en aplicaciones externas.
Una interfaz web interactiva en HTML, donde los usuarios pueden:
Subir imágenes de resonancia magnética.
Obtener resultados de clasificación con un porcentaje de confiabilidad del modelo.


Se debe colocar las imagenes de resonancia (para entrenamiento) clasificadas por carpetas en un archivo 'data' en la raiz del proyecto.

Instalación
-----------

# Crear entorno virtual para instalar dependencias a usar

```sh
python -m venv fastapi-env
```

# Activar entorno virtual

```sh
fastapi-env\Scripts\activate
```

# Instalar dependencias

```sh
pip install -r requirements.txt
```

# Ejecutar servidor ([powershell])

```sh
fastapi-env\Scripts\activate; uvicorn main:app --reload
```