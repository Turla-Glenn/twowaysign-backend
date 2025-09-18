import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("cnn8grps_rad1_model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ model.tflite saved!")

# Optional: Quantization (smaller + faster)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant = converter.convert()
with open("model_quant.tflite", "wb") as f:
    f.write(tflite_quant)

print("✅ model_quant.tflite saved!")
