import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Cargar los datos de carpeta con imagenes de tumores
# se usa la misma libreria de keras para colocar a escalas de grises
def load_imagenes(directorio, tamaño=(28, 28)):
    imagenes = []
    etiquetas = []
    for etiqueta, carpeta in enumerate(os.listdir(directorio)):
        ruta_carpeta = os.path.join(directorio, carpeta)
        for archivo in os.listdir(ruta_carpeta):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            imagen = load_img(ruta_archivo, target_size=tamaño, color_mode="grayscale")
            imagenes.append(img_to_array(imagen))
            etiquetas.append(etiqueta)
    return np.array(imagenes), np.array(etiquetas)

# metodo para preprocesar las imagenes
def preprocesar_imagenes(X, Y):
    #Colocar los datos en la forma correcta (1, 28, 28, 1)
    X = X.reshape(X.shape[0], 28, 28, 1).astype('float32') / 255
    Y = to_categorical(Y)
    return X, Y

path_dataset = 'data'

X, Y = load_imagenes(path_dataset)

# Divide en 80% para entrenamiento y 20% para prueba
X_entrenamiento, X_pruebas, Y_entrenamiento, Y_pruebas = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

X_entrenamiento, Y_entrenamiento = preprocesar_imagenes(X_entrenamiento, Y_entrenamiento)
X_pruebas, Y_pruebas = preprocesar_imagenes(X_pruebas, Y_pruebas)

#Codigo para mostrar imagenes del set :)
import matplotlib.pyplot as plt
filas = 2
columnas = 8
num = filas*columnas
imagenes = X_entrenamiento[0:num]
etiquetas = Y_entrenamiento[0:num]
fig, axes = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
for i in range(num):
     ax = axes[i//columnas, i%columnas]
     ax.imshow(imagenes[i].reshape(28,28), cmap='gray_r')
     ax.set_title('Label: {}'.format(np.argmax(etiquetas[i])))
plt.tight_layout()
plt.show()

#Aumento de datos
#Variables para controlar las transformaciones que se haran en el aumento de datos
#utilizando ImageDataGenerator de keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

rango_rotacion = 30
mov_ancho = 0.25
mov_alto = 0.25
#rango_inclinacion=15 #No uso este de momento
rango_acercamiento=[0.5,1.5]

datagen = ImageDataGenerator(
    rotation_range = rango_rotacion,
    width_shift_range = mov_ancho,
    height_shift_range = mov_alto,
    zoom_range=rango_acercamiento,
    #shear_range=rango_inclinacion #No uso este de momento
)

datagen.fit(X_entrenamiento)

clases = sorted(os.listdir(path_dataset))
NUMERO_DE_CLASES = len(clases)

filas = 4
columnas = 8
num = filas*columnas

# Gráfica ANTES
print('ANTES:\n')
fig1, axes1 = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
for i in range(num):
    ax = axes1[i//columnas, i%columnas]
    ax.imshow(X_entrenamiento[i].reshape(28,28), cmap='gray_r')
    etiqueta = np.argmax(Y_entrenamiento[i])  # Índice de la clase
    ax.set_title(f'Label: {etiqueta} ({clases[etiqueta]})') 
    #ax.set_title('Label: {}'.format(np.argmax(Y_entrenamiento[i])))
plt.tight_layout()
plt.show()

# Gráfica DESPUÉS
print('DESPUES:\n')
fig2, axes2 = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
for X, Y in datagen.flow(X_entrenamiento,Y_entrenamiento.reshape(Y_entrenamiento.shape[0], NUMERO_DE_CLASES),batch_size=num,shuffle=False):
     for i in range(0, num):
        ax = axes2[i//columnas, i%columnas]
        ax.imshow(X[i].reshape(28,28), cmap='gray_r')
        etiqueta = int(np.argmax(Y[i]))  # Índice de la clase
        ax.set_title(f'Label: {etiqueta} ({clases[etiqueta]})') 
        #ax.set_title('Label: {}'.format(int(np.argmax(Y[i]))))
     break
plt.tight_layout()
plt.show()


#Modelo!
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(NUMERO_DE_CLASES, activation="softmax")
])

#Compilación
modelo.compile(optimizer='adam', # utilizamos el optimizador Adam para mejorar la velocidad de convergencia
              loss='categorical_crossentropy',
              metrics=['accuracy'])

TAMANO_LOTE = 32
#Los datos para entrenar saldran del datagen, de manera que sean generados con las transformaciones que indicamos
data_gen_entrenamiento = datagen.flow(
    X_entrenamiento, Y_entrenamiento, batch_size=TAMANO_LOTE, shuffle=True
)

#Entrenamiento del modelo
print("Entrenando modelo...");
epocas=60
history = modelo.fit(
    data_gen_entrenamiento,
    epochs=epocas,
    batch_size=TAMANO_LOTE,
    validation_data=(X_pruebas, Y_pruebas),
    steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(TAMANO_LOTE))),
    validation_steps=int(np.ceil(len(X_pruebas) / float(TAMANO_LOTE)))
)

print("Modelo entrenado!")

#Exportar el modelo a un archivo .h5
carpeta_salida = "models_trained"
nombre_modelo = "trained_model_tumor.h5"

# Verificar si la carpeta existe, si no, crearla
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# Ruta completa para guardar el modelo
ruta_modelo = os.path.join(carpeta_salida, nombre_modelo)

# Guardar el modelo
modelo.save(ruta_modelo)