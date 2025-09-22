import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tf2onnx
import onnxruntime as ort

# 📌 Préparation des données
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    "data/",
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    "data/",
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# 📌 Création du modèle CNN
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3), name="input_layer"),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# 📌 Compilation
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 📌 Entraînement
model.fit(train_generator, validation_data=val_generator, epochs=10)

# 📌 Sauvegarde au nouveau format H5 compatible TF2
model.save("dog_vs_cat.h5", save_format="h5")  # format H5 standard TF2
print("✅ Model saved as cat_vs_dog.h5")

# 📌 Conversion en ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
with open("dog_vs_cat.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("✅ Model converted to cat_vs_dog.onnx")

# 📌 Test rapide ONNX
session = ort.InferenceSession("dog_vs_cat.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
result = session.run([output_name], {input_name: dummy_input})
print("✅ ONNX test successful, output =", result)






