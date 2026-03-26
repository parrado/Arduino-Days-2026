import numpy as np
from tensorflow import keras
from spectral_features import generate_features
import matplotlib.pyplot as plt



# Path to the JSON files containing accelerometer data for training
jsonPath="..//gestos_nano_esp32-export//training"

# 1. Generate features and labels from the JSON files
train_input, train_output = generate_features(jsonPath)



# 2. Build the model
model = keras.Sequential([
    keras.Input(shape=(27,)),  # Input layer with 27 features
    keras.layers.Dense(20, activation='relu'), # Hidden layer with 20 neurons and 'relu' activation
    keras.layers.Dense(10, activation='relu'), # Hidden layer with 10 neurons and 'relu' activation
    # Output layer with 4 neurons and 'softmax' activation
    keras.layers.Dense(4, activation='softmax')
])

# 3. Compile the model
# Use 'categorical_crossentropy' loss for multi-class problems with one-hot labels
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
history=model.fit(train_input, train_output, epochs=20, batch_size=16)

# 5. Save the trained model to a file
model.save('gesture_model.h5')

# Assuming history is from model.fit()
acc = history.history['accuracy']

loss = history.history['loss']


# Plotting Accuracy
plt.figure(1)
plt.plot(acc)
plt.title('Accuracy')
plt.xlabel('Epoch')  # Add x-axis label
plt.ylabel('Accuracy')  # Add y-axis label
plt.grid()  # Add grid for better visibility




# Plotting Loss
plt.figure(2)
plt.plot(loss)
plt.title('Loss')
plt.xlabel('Epoch')  # Add x-axis label
plt.ylabel('Loss')  # Add y-axis label
plt.grid()  # Add grid for better visibility    
plt.show()
print(len(loss))








