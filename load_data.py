from tensorflow.keras.preprocessing.image import ImageDataGenerator

# normalize images
train_datagen = ImageDataGenerator(rescale=1./255)

# load training data
train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

print("Classes:", train_data.class_indices)