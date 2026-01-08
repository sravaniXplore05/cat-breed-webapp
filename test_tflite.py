import tensorflow as tf

interpreter = tf.lite.Interpreter(
   model_path="models/cat_breed_model_v3.tflite"
)
interpreter.allocate_tensors()

print("TFLite model loaded successfully")
