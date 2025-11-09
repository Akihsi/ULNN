import sys
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "resnet50_ai_detection.h5"
IMG_SIZE = (224, 224)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img.astype("float32"), axis=0)
    return img

def run_inference(path):
    img = preprocess(path)
    
    # model outputs [prob_real, prob_ai]
    prediction = model.predict(img, verbose=0)[0]
    prob_real = float(prediction[0])
    prob_ai = float(prediction[1])

    print(f"\nRAW MODEL OUTPUT: {prediction}")

    if prob_real > prob_ai:
        verdict = "Real Image"
        confidence = prob_real * 100
    else:
        verdict = "AI Generated"
        confidence = prob_ai * 100

    print(f"Verdict: {verdict}")
    print(f"Confidence: {confidence:.2f}%\n")

    return {"verdict": verdict, "confidence": confidence}

if __name__ == "__main__":
    img_path = sys.argv[1]
    run_inference(img_path)
