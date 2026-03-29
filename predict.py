import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# load trained model
model = load_model("pneumonia_model.h5")

# path to test image (change this!)
img_path = "dataset/test/NORMAL/IM-0001-0001.jpeg"

# load image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# predict
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")