import os
import tensorflow as tf

# Define dataset paths
dataset_dir = "../data/BreaKHis-400X"
train_dir= os.path.join(dataset_dir,"train")
test_dir= os.path.join(dataset_dir,"test")

"""
CNN models like ResNet or MobileNet often require images of a fixed size.
Here, we resize all images to 224x224
 """
# Image dimensions
img_height =224
img_width = 224
batch_size = 32  #specifies how many images to process at a time during training/testing.


"""
The ImageDataGenerator handles preprocessing and augmentation of the dataset.
"""
#Preprocessing for training and validation datasets
train_datagen= tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,       # Normalize pixel values to range [0, 1] for faster training
    rotation_range=20,       # Randomly rotate images up to 20 degrees
    width_shift_range =0.2,  # Randomly shift images horizontally by 20%
    height_shift_range=0.2,  # Randomly shift images vertically by 20%
    shear_range=0.2,         # Apply shearing transformations.Augmentation, helps the model generalize better
    zoom_range=0.2,          # Randomly flip images horizontally
    horizontal_flip=True,    # Randomly flip images horizontally
    validation_split=0.2     # Reserve 20% of training data for validation
)
"""
Preprocessing for test dataset (no augmentation). No augmentation for test data; only normalization.
"""
test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
    rescale =1.0/ 255.0     #   Only normalize test data
)

"""
Load training dataset.
Use the .flow_from_directory() method to load images from directories.
Loads images from train/ and applies augmentation for the training subset.
class_mode='binary' maps benign to 0 and malignant to 1.
"""
train_data= train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height,img_width), # Resize all images to 224x224
    batch_size= batch_size,
    class_mode= 'binary',                # Binary classification (benign vs malignant)
    subset= 'training'                   # Use training subset
)

"""
Load validation dataset.
Loads the validation subset of train/
"""
val_data =train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'                 # Use validation dataset
)

"""
Load testing dataset
Loads test data without augmentation, only normalization.
"""
test_data = test_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

#Print dataset details
print("Training data loaded with",train_data.samples,"samples.")
print("Validation data loaded with", val_data.samples, "samples.")
print("Test data loaded with", test_data.samples, "samples.")


