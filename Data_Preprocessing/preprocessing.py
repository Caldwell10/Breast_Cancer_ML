import os
import tensorflow as tf

# Dataset location
dataset_dir = "/Users/caldwellwachira/Downloads/Breast_Cancer_Dataset"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

# Image dimensions and batch size
img_height = 224
img_width = 224
batch_size = 32

# ImageDataGenerator for training and validation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,          # Normalize pixel values
    rotation_range=20,            # Randomly rotate images
    width_shift_range=0.2,        # Randomly shift images horizontally
    height_shift_range=0.2,       # Randomly shift images vertically
    shear_range=0.2,              # Randomly shear images
    zoom_range=0.2,               # Randomly zoom in/out on images
    horizontal_flip=True,         # Randomly flip images horizontally
    brightness_range=[0.8, 1.2],  # Simulate lighting conditions
    fill_mode="nearest",          # Handle empty pixels after transformations
    validation_split=0.2          # Reserve 20% of training data for validation
)

# ImageDataGenerator for testing (no augmentation)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0           # Normalize pixel values
)

# Load training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),  # Resize images to the target dimensions
    batch_size=batch_size,
    class_mode="binary",                  # Binary classification (e.g., benign vs malignant)
    subset="training"                     # Use the training subset
)

# Load validation data
val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),  # Resize images to the target dimensions
    batch_size=batch_size,
    class_mode="binary",                  # Binary classification
    subset="validation"                   # Use the validation subset
)

# Load testing data
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),  # Resize images to the target dimensions
    batch_size=batch_size,
    class_mode="binary"                   # Binary classification
)

# Print dataset details
print(f"Training data loaded with {train_data.samples} samples.")
print(f"Validation data loaded with {val_data.samples} samples.")
print(f"Test data loaded with {test_data.samples} samples.")
