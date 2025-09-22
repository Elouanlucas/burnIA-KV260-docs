import tensorflow as tf
import tf2onnx
from keras.layers import InputLayer

# Charger le modèle en ignorant batch_shape
model = tf.keras.models.load_model("dog_vs_cat.h5", compile=False, custom_objects={"InputLayer": InputLayer})

# Conversion en ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

with open("dog_vs_cat.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ Conversion completed: cat_vs_dog.onnx created")
