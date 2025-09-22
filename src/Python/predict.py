from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Charger ton modèle sauvegardé
model = load_model("dog_vs_cat.h5")

def predict_image(img_path):
    # Prépare l’image
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Faire la prédiction
    prob = model.predict(img_array)[0][0]

    if prob > 0.5:
        print(f"{img_path} 🐶 → It’s a dog (confidence = {prob:.2f})")
    else:
        print(f"{img_path} 🐱 → It’s a cat (confidence = {1-prob:.2f})")


# --- TESTS (à la fin du fichier) ---
import os

test_folder = "test/"

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    try:
        result = predict_image(img_path)
        print(f"{img_name} -> {result}")
    except Exception as e:
        print(f"Error with {img_name}: {e}")
