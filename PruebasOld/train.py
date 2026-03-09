import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
import os

# ======================
# CONFIGURACIÓN
# ======================
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 30

DATASET_DIR = "dataset_processed"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "symbol_model_efficientnet.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# ======================
# DATA GENERATORS
# ======================
datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print("Clases detectadas:", train_gen.class_indices)

# ======================
# BASE MODEL (PRETRAINED)
# ======================
input_layer = layers.Input(shape=(96,96,1))

# convertir grayscale -> RGB
x = layers.Concatenate()([input_layer, input_layer, input_layer])

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_tensor=x
)

base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

# ======================
# CLASIFICADOR
# ======================
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# CALLBACKS
# ======================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

# ======================
# ENTRENAMIENTO
# ======================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# ======================
# GUARDAR MODELO
# ======================
model.save(MODEL_PATH)
print(f"\nModelo guardado en: {MODEL_PATH}")

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

val_gen.reset()
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)