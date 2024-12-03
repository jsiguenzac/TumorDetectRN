import numpy as np
import cv2
import tensorflow as tf
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import regularizers
"""
Definición de cada una de clases, existen 4 clases de distintas, tenemos 
3 tipos de cáncer distintos 
de cáncer y una clase que representa la ausencia de cáncer
"""
# Cargar y procesar las etiquetas
labels = ['no_tumor','glioma_tumor','meningioma_tumor','pituitary_tumor']
#Variables donde se guardaran las imagenes y sus labels correspondientes"

X = []
Y = []
image_size = 224

# Cargar imágenes de entrenamiento
for i in labels:
    # Combinar la ruta base con la etiqueta
    folderPath = os.path.join('/folders/1xvX6WGmo9ojLxe0O66oy6ZBPfqMDgh3K', i)
    print('FilePath:', folderPath)
    # Iterar a través de los archivos en la carpeta
    for j in tqdm(os.listdir(folderPath), desc=f"Procesando etiqueta {i}"):
        img_path = os.path.join(folderPath, j)
        
        # Leer y procesar la imagen
        img = cv2.imread(img_path)
        if img is not None:  # Verifica si la imagen se cargó correctamente
            img = cv2.resize(img, (image_size, image_size))
            X.append(img)
            Y.append(i)

# Cargar imágenes de prueba
for i in labels:
    # Construir la ruta a la carpeta de la etiqueta
    folderPath = os.path.join('/u/2/folders/15tutmIreaM9rbPt8Afbmvwow6po0dhra', i)
    
    # Verificar si la carpeta existe
    if not os.path.exists(folderPath):
        print(f"Advertencia: La carpeta {folderPath} no existe. Saltando...")
        continue
    
    # Leer y procesar cada imagen en la carpeta
    for j in tqdm(os.listdir(folderPath), desc=f"Procesando etiqueta {i}"):
        img_path = os.path.join(folderPath, j)
        
        # Leer la imagen y verificar que se cargó correctamente
        img = cv2.imread(img_path)
        if img is None:
            print(f"Advertencia: No se pudo leer la imagen {img_path}. Saltando...")
            continue
        
        # Redimensionar la imagen y agregarla a la lista
        img = cv2.resize(img, (image_size, image_size))
        X.append(img)
        Y.append(i)

# Convertir listas a arrays de NumPy
X = np.array(X, dtype='float32')  # Asegurarse de que sean del tipo correcto
Y = np.array(Y)

# Definir función para aumentar datos
def aumentar_datos(X, Y, n):
    assert isinstance(X, np.ndarray), "X debe ser un array de numpy"
    assert isinstance(Y, np.ndarray), "Y debe ser un array de numpy"
    assert len(X.shape) == 4, "X debe ser un tensor 4D con dimensiones (batch, height, width, channels)"
    assert len(Y.shape) == 1, "Y debe ser un tensor 1D con dimensiones (batch,)"
    assert len(X) == len(Y), "X e Y deben tener la misma longitud"
    assert isinstance(n, int) and n > 0, "n debe ser un entero positivo"

    X_aug = []
    Y_aug = []

    for i in range(len(X)):
        imagen = X[i]
        etiqueta = Y[i]

        for j in range(n):
            angulo = np.random.randint(-15, 15)
            M = cv2.getRotationMatrix2D((imagen.shape[1] // 2, imagen.shape[0] // 2), angulo, 1)
            imagen_rotada = cv2.warpAffine(imagen, M, (imagen.shape[1], imagen.shape[0]))

            tx = np.random.randint(-10, 10)
            ty = np.random.randint(-10, 10)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            imagen_traslada = cv2.warpAffine(imagen_rotada, M, (imagen.shape[1], imagen.shape[0]))

            if np.random.random() < 0.5:
                imagen_reflejo = cv2.flip(imagen_traslada, 1)
            else:
                imagen_reflejo = imagen_traslada

            alpha = np.random.uniform(0.5, 2.0)
            beta = np.random.randint(-30, 30)
            imagen_bc = cv2.convertScaleAbs(imagen_reflejo, alpha=alpha, beta=beta)

            X_aug.append(imagen_bc)
            Y_aug.append(etiqueta)

    return np.array(X_aug), np.array(Y_aug)

# División de datos en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# Preprocesamiento de imágenes
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

# Convertir etiquetas a one-hot encoding
Y_train_one_hot = to_categorical(Y_train, num_classes=4)
Y_test_one_hot = to_categorical(Y_test, num_classes=4)

# Construcción del modelo basado en ResNet50
base_model = ResNet50(weights='imagenet', include_top=False)

# Congelar capas base
for layer in base_model.layers:
    layer.trainable = False

# Definir capas superiores del modelo
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy', tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]
)

# Descongelar algunas capas para fine-tuning
for layer in base_model.layers:
    if "conv5_block3" in layer.name:
        layer.trainable = True

# Entrenamiento del modelo
weights_path = "/content/drive/MyDrive/tfg/pesos/weights.h5"
num_epochs = 100
batch_size = 32

history = model.fit(
    X_train, Y_train_one_hot,
    validation_data=(X_test, Y_test_one_hot),
    epochs=num_epochs,
    batch_size=batch_size
)

# Guardar modelo entrenado
model.save(weights_path)

# Visualización de métricas
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
false_positive = history.history['false_positives']
val_false_positive = history.history['val_false_positives']
false_negative = history.history['false_negatives']
val_false_negative = history.history['val_false_negatives']

fig, axs = plt.subplots(4, 1, figsize=(8, 12))
axs[0].plot(acc, label='Training Accuracy')
axs[0].plot(val_acc, label='Validation Accuracy')
axs[0].legend(loc='lower right')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Training and Validation Accuracy')

axs[1].plot(loss, label='Training Loss')
axs[1].plot(val_loss, label='Validation Loss')
axs[1].legend(loc='upper right')
axs[1].set_ylabel('Cross Entropy')
axs[1].set_title('Training and Validation Loss')

axs[2].plot(false_positive, label='Training False Positives')
axs[2].plot(val_false_positive, label='Validation False Positives')
axs[2].legend(loc='upper right')
axs[2].set_ylabel('False Positives')
axs[2].set_title('False Positives')

axs[3].plot(false_negative, label='Training False Negatives')
axs[3].plot(val_false_negative, label='Validation False Negatives')
axs[3].legend(loc='upper right')
axs[3].set_ylabel('False Negatives')
axs[3].set_title('False Negatives')
axs[3].set_xlabel('Epochs')

plt.show()

# Evaluar modelo en datos de prueba
preds = model.evaluate(X_test, Y_test_one_hot)
print("Loss =", preds[0])
print("Test Accuracy =", preds[1])