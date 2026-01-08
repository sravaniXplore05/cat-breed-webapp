import tensorflow as tf

H5_MODEL_PATH = r"C:\Users\K SAI SRAVANI\OneDrive\Desktop\cat_breed_prediction\mobilenetv2_cat_model_v3.h5"
TFLITE_MODEL_PATH = r"C:\Users\K SAI SRAVANI\OneDrive\Desktop\cat_breed_webapp\models\cat_breed_model_v3.tflite"

# Load Keras model
model = tf.keras.models.load_model(H5_MODEL_PATH)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model converted successfully")
