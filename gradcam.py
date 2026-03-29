import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------- LOAD IMAGE ----------------
img_path = "dataset/test/PNEUMONIA/person1674_virus_2890.jpeg"

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# ---------------- LOAD MODEL ----------------
model = load_model("pneumonia_model.h5")

# 🔥 FORCE BUILD (CRITICAL)
preds = model.predict(img_array)

# ---------------- ACCESS BASE MODEL ----------------
base_model = model.layers[0]  # MobileNetV2

# LAST CONV LAYER
last_conv_layer = base_model.get_layer("Conv_1")

# ---------------- CREATE GRAD MODEL (SAFE VERSION) ----------------
grad_model = tf.keras.models.Model(
    inputs=model.layers[0].input,   # 🔥 USE BASE MODEL INPUT
    outputs=[
        last_conv_layer.output,
        model.layers[-1](base_model.output)  # 🔥 MANUAL FORWARD PASS
    ]
)

# ---------------- GRAD-CAM ----------------
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# normalize
heatmap = np.maximum(heatmap, 0)
heatmap /= (np.max(heatmap) + 1e-8)

# ---------------- SUPERIMPOSE ----------------
img_cv = cv2.imread(img_path)
img_cv = cv2.resize(img_cv, (224, 224))

heatmap = cv2.resize(heatmap, (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

# ---------------- DISPLAY ----------------
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Grad-CAM: Pneumonia Detection")
plt.show()