from PIL import Image
import numpy as np
import onnxruntime as ort
import os

# === CONFIGURATION ===
model_path = "/home/ubuntu/models/dog_vs_cat.onnx"
images_folder = "/home/ubuntu/images_to_classify/"
classes = ["cat", "dog"]
input_size = (128, 128)  # Taille attendue par le modèle (H, W)

# === VERIFIER LE DOSSIER ===
if not os.path.exists(images_folder):
    print(f"The folder {images_folder} does not exist.")
    exit()

# === CREER LA SESSION ONNX ===
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# === FONCTION DE PREPROCESS ===
def preprocess_image(img_path, target_size):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_data = np.array(img).astype(np.float32)
    img_data = np.expand_dims(img_data, axis=0)  # Ajouter dimension batch
    return img_data  # NHWC : [1, H, W, C]

# === BOUCLE SUR LES IMAGES ===
results = {}
for filename in os.listdir(images_folder):
    file_path = os.path.join(images_folder, filename)
    if not os.path.isfile(file_path):
        continue
    try:
        img_data = preprocess_image(file_path, input_size)
        pred = session.run(None, {input_name: img_data})[0]
        predicted_class = classes[np.argmax(pred)]
        results[file_path] = predicted_class
    except Exception as e:
        print(f"Erreur pour {filename}: {e}")

# === AFFICHER LES RESULTATS ===
print("\n=== INFERENCE RESULTS ===")
for img_path, pred_class in results.items():
    print(f"{os.path.basename(img_path)} --> {pred_class}")
