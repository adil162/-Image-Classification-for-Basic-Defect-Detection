# Importing all the required libraries
import numpy as np #Using for Numerical Operations
import matplotlib.pyplot as plt #Using for Plotting
import tensorflow as tf #Using for Deep Learning
from tensorflow.keras.datasets import cifar10 #using the cifar-10 dataset
from tensorflow.keras.utils import to_categorical #Using one-hot encoded labels for classification tasks. 
# One-hot encoding changes the label into a vector (list of 0s and 1s) that makes it clear which class is correct
import os #Using for file operations
import pickle #using for saving and loading python objects
from sklearn.metrics import classification_report, confusion_matrix #using for evaluating the performance of a classification model
import seaborn as sns #using for visualizing the confusion matrix



def load_cifar10_batch(batch_filename):
    """
    This code defines a function that loads a batch of the CIFAR-10 dataset from a file.
    The function takes a file name as input and returns the images and labels as numpy arrays.
    The images are stored as a 4D array of shape (batch_size, height, width, channels)
    and the labels are stored as a 1D array of shape (batch_size,).
    The function uses the pickle library to read the binary data from the file.
    """
    #This function takes a file name of a batch in the CIFAR-10 dataset and load the data from the file
    with open(batch_filename, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
            #We use pickle to load the binary data in binary mode from the file.

        X = dict[b'data']
            #The data is stored in a dictionary with the key 'data'

        Y = dict[b'labels']
            #The labels are stored in a dictionary with the key 'labels'

        X = X.reshape(10000, 3, 32, 32)
            #The data is stored as a 1D array, so we reshape it to 4D

        X = X.transpose(0, 2, 3, 1)
            # We transpose the array to change the order of the axes.
            # We do this because the model is expecting the input to be in (batch, height, width, channels) format.
            # The original data is stored as (batch, channels, height, width).
            # We change it to (batch, height, width, channels) so that the model can understand it.
            # This is necessary because the model is not flexible enough to accept the data in a different format.
            # Think of it like this:
            # Original: (batch, channels, height, width) = (10000, 3, 32, 32)
            # Transposed: (batch, height, width, channels) = (10000, 32, 32, 3)
            # Now the model can understand it as (batch, height, width, channels)
            # But the CIFAR-10 data is stored in (batch, channels, height, width) format.

        Y = np.array(Y)
            # Convert the labels to a numpy array

        return X, Y 
            # Return the images and labels

def load_cifar10_data(data_dir):
    """
    This function loads the CIFAR-10 dataset from a directory.
    The dataset is divided into 5 batches, and each batch is a separate file.
    The function reads each file, extracts the images and labels,
    and returns them as a single array of images and a single array of labels.
    """
    
    # Create empty lists to store the images and labels
    x_train = []
    y_train = []
    
    for i in range(1, 6):
    # Loop through the 5 batches
    # The batches are numbered from 1 to 5
    # We loop through the range 1 to 5 (inclusive)
    # For each iteration, we construct the filename for the current batch
    # The filename is a combination of the directory and the batch number
    # For example, if the directory is '/path/to/data', the filename for the first batch would be '/path/to/data/data_batch_1'

        # Construct the filename for the current batch
        batch_filename = os.path.join(data_dir, f'data_batch_{i}')
        
        # Load the current batch
        X, Y = load_cifar10_batch(batch_filename)
        
        # Append the current batch to the lists
        x_train.append(X)
        y_train.append(Y)
    
    # Concatenate the lists into single arrays means combining all the batches into one array
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    
    # Load the test batch
    # The test batch is a single file that contains all the test images and labels
    # The return statement returns the training and test data as two separate tuples
    # The first tuple contains the training images and labels
    # The second tuple contains the test images and labels
    x_test, y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return (x_train, y_train), (x_test, y_test)


# Path to extracted CIFAR-10
data_dir = os.path.expanduser("~/.keras/datasets/cifar-10-batches-py")

# Load the data manually
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

# Define class labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

# Simulate: airplane = 0 (non-defective), truck = 9 (defective)
non_defective_class = 0  # airplane
defective_class = 9      # truck


# Filter data for two classes only
def filter_classes(x, y, class1, class2):
    """
    This function filters the data for two specific classes.
    The purpose of this function is to select only the images and labels
    that correspond to the two classes of interest.
    The function takes in the original data, the class labels of the two classes,
    and returns the filtered data.

    Parameters:
    x (numpy array): The original images
    y (numpy array): The original labels
    class1 (int): The first class label
    class2 (int): The second class label

    Returns:
    x (numpy array): The filtered images
    y (numpy array): The filtered labels
    """

    # Find the indices of the images that correspond to either class1 or class2
    idx = np.where((y == class1) | (y == class2))[0]
    
    # Select only the images and labels at these indices
    x = x[idx]
    y = y[idx]
    
    # Replace the labels with 0 and 1
    # 0 corresponds to class1(Airplane), and 1(Truck) corresponds to class2
    y = np.where(y == class1, 0, 1)
    
    y = np.where(y == class1, 0, 1)  # 0 = non-defective, 1 = defective
    return x, y

# Filter the data for the two classes of interest
x_train, y_train = filter_classes(x_train, y_train, non_defective_class, defective_class)
x_test, y_test = filter_classes(x_test, y_test, non_defective_class, defective_class)


# Convert the training and testing image data to float32 type and normalize them to the range [0, 1]
# This ensures that the model receives inputs in the expected format and range, improving performance
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# Print information about the dataset to understand its structure and distribution
print("Training samples:", x_train.shape[0])  # Number of training samples
print("Testing samples:", x_test.shape[0])    # Number of testing samples
print("Image shape:", x_train.shape[1:])      # Shape of each image
# Distribution of labels in the training set
print("Label distribution (train):", np.bincount(y_train)) 
# Distribution of labels in the testing set
print("Label distribution (test):", np.bincount(y_test))   


# --------------- DATA AUGMENTATION -----------------
# Data augmentation is a technique used to artificially increase the size of the training dataset
# by applying random transformations to the images. This helps the model generalize better
# to unseen data and improves its performance.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an instance of ImageDataGenerator with specified data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=15,          # Randomly rotate images by up to 15 degrees
    width_shift_range=0.1,      # Randomly shift images horizontally by 10% of total width
    height_shift_range=0.1,     # Randomly shift images vertically by 10% of total height
    horizontal_flip=True,       # Randomly flip images horizontally
    validation_split=0.2        # Reserve 20% of the data for validation
)

# Create a data generator for training data with augmentation
train_generator = datagen.flow(
    x_train, y_train, 
    subset='training',         # Use the training data subset
    batch_size=64              # Specify the batch size for training
)

# Create a data generator for validation data with augmentation
val_generator = datagen.flow(
    x_train, y_train, 
    subset='validation',       # Use the validation data subset
    batch_size=64              # Specify the batch size for validation
)


# --------------- MODEL DEFINITION ------------------
# Create a model with a sequence of layers
model = tf.keras.Sequential()

# The first layer is a convolutional layer with 32 filters
# The filters have a size of 3x3 and are activated with the ReLU (Rectified Linear Unit) function
# ReLU (Rectified Linear Unit) is an activation function that outputs 0 if the input is negative, and the input value if the input is positive.
# It is used to introduce non-linearity in the model and to speed up the training process by avoiding the vanishing gradients problem
# The input shape is 32x32x3, which is the size of the images
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# The second layer is a max pooling layer with a pool size of 2x2
# This reduces the spatial dimensions of the output volume to half
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# The third layer is a convolutional layer with 64 filters
# The filters have a size of 3x3 and are activated with the ReLU function
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

# The fourth layer is a max pooling layer with a pool size of 2x2
# This reduces the spatial dimensions of the output volume to half
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# The fifth layer is a flatten layer, which flattens the output volume
# into a 1D feature vector
model.add(tf.keras.layers.Flatten())

# The sixth layer is a dense layer with 64 units
# The units are activated with the ReLU function
model.add(tf.keras.layers.Dense(64, activation='relu'))

# The seventh layer is a dense layer with 1 unit
# The unit is activated with the sigmoid function, which is used for binary classification
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model with the adam optimizer and binary cross-entropy loss
# The metrics are set to accuracy, which is used to evaluate the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# --------------- EARLY STOPPING CALLBACK -----------
# This callback stops training when a monitored metric has stopped improving.
# Here, it monitors the 'val_loss' (validation loss) and will stop the training
# if it hasn't improved for 5 consecutive epochs (indicated by 'patience=5').
# 'restore_best_weights=True' ensures that the model weights are set to the 
# best weights observed during the training process.
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --------------- TRAIN THE MODEL --------------------
# This step trains the model on the training data and evaluates it on the validation data.
# The 'fit' method takes in the training data, validation data, number of epochs to train for,
# and any callbacks to use during training. The 'callbacks' parameter is used to pass in the
# 'early_stop' callback, which will stop training if the model's performance on the validation
# set hasn't improved for 5 consecutive epochs. The model is trained for 20 epochs.
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# --------------- EVALUATE THE MODEL -----------------
# This code block evaluates the trained model on the test dataset.
# It calculates the loss and accuracy of the model on unseen data (x_test and y_test).
# The 'evaluate' method returns two values: test_loss and test_acc.
# The 'test_loss' is the value of the loss function on the test data, and 'test_acc' is the accuracy.
# Printing the test accuracy helps us understand how well the model performs on new data.
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n🧪 Test accuracy: {test_acc:.4f}")

# --------------- PREDICTIONS FOR REPORT -------------
# The 'predict' method is used to make predictions on the test data.
# The model is given the test data (x_test) and it outputs the predicted probabilities.
# The predicted probabilities are then rounded to 0 or 1 to get the predicted class labels.
# This is done by comparing the predicted probabilities with 0.5, which is the threshold for the sigmoid function.
# If the predicted probability is greater than 0.5, the predicted class label is set to 1 (Defective).
# If the predicted probability is less than or equal to 0.5, the predicted class label is set to 0 (Non-Defective).
y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

# Classification Report
# This code block prints a classification report for the model on the test data.
# The classification report shows the precision, recall, f1-score, and support for each class.
# The target_names parameter is used to specify the class names in the report.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Defective', 'Defective']))

# Confusion Matrix
# A confusion matrix is a table that is used to evaluate the performance of a classification model.
# It is a square matrix where the number of rows and columns are equal to the number of classes in the classification problem.
# The entry in the ith row and jth column is the number of samples that are actually in class i but are predicted to be in class j.
# In this case, the confusion matrix is a 2x2 matrix, where the rows and columns represent the true and predicted labels, respectively.
# The confusion matrix is defined as follows:
# [[True Negatives, False Positives], [False Negatives, True Positives]]
# Where TN is the number of samples that are actually negative and are predicted to be negative,
# FP is the number of samples that are actually negative but are predicted to be positive,
# FN is the number of samples that are actually positive but are predicted to be negative,
# TP is the number of samples that are actually positive and are predicted to be positive.
# The confusion matrix is calculated using the sklearn.metrics.confusion_matrix function.
cm = confusion_matrix(y_test, y_pred)

# This line of code creates a new figure for plotting with a specified size of 6 inches by 5 inches.
plt.figure(figsize=(6, 5))
# This code creates a heatmap visualization of the confusion matrix.
# The heatmap is a 2D representation of the matrix where the color of each cell
# represents the number of samples that fall into that cell.
# The x-axis represents the predicted labels and the y-axis represents the actual labels (true labels).
# The number in each cell is the number of samples that fall into that cell.
# For example, if the top-left cell has a value of 10, it means that 10 samples
# were predicted as 'Non-Defective' and were actually 'Non-Defective' in reality.
# The color of the cell will be a shade of blue, with darker shades meaning more samples.
sns.heatmap(cm, annot=True, cmap='Blues',
            xticklabels=['Non-Defective', 'Defective'],
            yticklabels=['Non-Defective', 'Defective'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# --------------- PLOT TRAINING HISTORY ----------------
plt.figure(figsize=(12, 5))

# Plot the accuracy and loss over epochs
# This will show us how the model is doing over time
# We plot the training and validation accuracy and loss
# This will give us an idea if the model is overfitting or not
plt.subplot(1, 2, 1)
# The training accuracy is the accuracy on the training data
# The validation accuracy is the accuracy on the validation data
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# The title of the plot is "Accuracy Over Epochs"
plt.title('Accuracy Over Epochs')
# The x-axis is the epoch number
plt.xlabel('Epoch')
# The y-axis is the accuracy
plt.ylabel('Accuracy')
# The legend will show the labels for the two plots
plt.legend()

# The second subplot is for the loss
plt.subplot(1, 2, 2)
# Plot the loss over epochs for the training data
plt.plot(history.history['loss'], label='Train Loss')
# Plot the loss over epochs for the validation data
plt.plot(history.history['val_loss'], label='Val Loss')
# The title of the plot is "Loss Over Epochs"
plt.title('Loss Over Epochs')
# The x-axis is the epoch number
plt.xlabel('Epoch')
# The y-axis is the loss
plt.ylabel('Loss')
# The legend will show the labels for the two plots
plt.legend()

# The purpose of plotting loss over epochs is to see how the model is improving over time.
# We want to see if the model is overfitting or not.
# If the model is overfitting, we will see that the training loss will continue to decrease while the validation loss will increase.
# If the model is not overfitting, we will see that both the training and validation loss will continue to decrease.
# By plotting the loss over epochs, we can see how the model is doing over time and make adjustments to the model if necessary.

# Show the plots
plt.show()

# --------------- VISUALIZE SOME TEST PREDICTIONS ----------------
# The purpose of visualizing some test predictions is to see how well the model is doing on the test data.
# We want to see if the model is correctly predicting the labels for the test data.
# To do this, we will visualize some of the test data along with the predicted labels.
# We will also show the true labels for comparison.
# This will help us see if the model is correctly predicting the labels for the test data.

# First, we will get the predicted labels for the test data.
# We will use the model to make predictions on the test data.
# We will get the predicted probabilities and then convert them to labels.
# We will then compare the predicted labels with the true labels.

# Next, we will create a figure with subplots.
# Each subplot will show one of the test images along with the predicted label and the true label.
# We will use the imshow function to display the images.
# We will set the title of each subplot to show the predicted label and the true label.
# We will also set the axis to off to hide the axis labels.

# Finally, we will show the plots.
plt.figure(figsize=(15, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i])
    plt.title(f"Pred: {'Defective' if y_pred[i] else 'Non-Defective'}\nTrue: {'Defective' if y_test[i] else 'Non-Defective'}")
    plt.axis('off')
plt.show()

# --------------- SAVE THE MODEL ----------------------
model.save('defect_detection_cnn_model.h5')
print("✅ Model saved as defect_detection_cnn_model.h5")