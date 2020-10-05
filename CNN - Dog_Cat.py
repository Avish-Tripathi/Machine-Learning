# Convolutional Neural Network

# Building a CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step-1 Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step-2 Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding another convolution layer (for better accuracy)
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step-3 Flattening
classifier.add(Flatten())

# Step-4 Fully connected ANN
classifier.add(Dense(activation = 'relu', output_dim = 128 ))
classifier.add(Dense(activation = 'sigmoid', output_dim = 1 ))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss  = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting CNN to dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=1,
        validation_data=test_set,
        validation_steps=2000)

# Making new single predictions
import numpy as np
from keras.preprocessing import image
test_img = image.load_img('single_prediction/image.jpg', target_size = (64, 64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
result = classifier.predict(test_img)
training_set.class_indices
if result[0][0]==1:
    prediction = 'This is a Dog'
else:
    prediction = 'This is a Cat'
