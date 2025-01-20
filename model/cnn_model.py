import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Import preprocesssed datasets
from Data_Preprocessing.preprocessing import train_data, val_data, test_data

# Path to save model weights in iCloud
icloud_save_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Models/best_model.weights.h5")

# === Define CNN Model ===
def build_model(input_shape=(224,224,3)):
    """
       Builds a Convolutional Neural Network (CNN) model for binary classification.

       Args:
           input_shape: Shape of the input images (height, width, channels).

       Returns:
           A compiled Keras model ready for training.


    """
    model = tf.keras.Sequential([
            # Convolutional Layer 1
            tf.keras.layers.Conv2D(32,(3,3), activation='relu',input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2,2)),

             # Convolutional Layer 2
             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
             tf.keras.layers.MaxPooling2D((2, 2)),

             # Convolutional Layer 3
             tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
             tf.keras.layers.MaxPooling2D((2, 2)),

             # Flatten and Fully Connected Layers
             tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(128, activation='relu'),
             tf.keras.layers.Dropout(0.5),   # Prevent overfitting

             # Output Layer for Binary Classification
             tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    return model

# Build the model
model = build_model()
model.summary()

# ===Compile the Model ===
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# === Callbacks ===
callbacks =[
    tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint( filepath=icloud_save_path, save_best_only=True, save_weights_only=True),
]

# === Train the model ===
history= model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=callbacks,
)

# === Evaluate the model ===
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# === Visualize Training Perfomance ===
plt.plot(history.history['accuracy'], label ='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Plot training and validation loss
plt.plot(history.history['loss'], label ='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# === Save the Final Model ===
model.save('breast_cancer_cnn_model.keras')
print("Model saved as 'breast_cancer_cnn_model.keras")