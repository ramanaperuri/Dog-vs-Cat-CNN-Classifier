# Dogs-vs-Cats-CNN-Classifier
This Github repo contains the folder, dogsncats, used to deploy a Dogs-vs-Cats classifier to Heroku 
and an unused folder with a transfer learning model that didn't get deployed due to server space limitations.

- You can find the Heroku app here: https://dogsncats.herokuapp.com/

The assignment or challenge for this classifier was to build a Convolutional Neural Network able to classify cat and dog images. And this
based on the Kaggle Dogs vs Cats dataset which contains 25000 labeled images and 12500 unlabeled images.

- The Kaggle dataset : https://www.kaggle.com/c/dogs-vs-cats

My original idea was to make the classifier app with 2 CNN models and compare them:
- 1 'home made' trained from scratch on the Kaggle dataset images 
and 
- 1 finetuned via transfer learning on the Kaggle dataset, based on the MobileNetV2 CNN 
The MobileNetV2 was originally trained on 1000+ classes and millions of images in the ImageNet dataset.

The python code for the 'home made' CNN can be found in the repo below under 'Test CatsvsDogs.py, the model under is saved as 'model_epochs.h5'.
 - https://github.com/kristof-becode/Deep-Learning/tree/master/Keras/Kaggle%20Dogs%20vs%20Cats
 
 The python code for the feature extraction and finetuning for the transfer learning can be found in 'Test CatsvsDogs MobileNet V2.py' and the trained models as the.h5 files. The finetuned model is the one to use.
- https://github.com/kristof-becode/Deep-Learning/tree/master/Keras/Transfer%20Learning%20with%20MobileNetV2

Due to Heroku server space limitations I couldn't make an app with both models so I decided to include only my 'home made' CNN.
- images were seperated by label and read from directory, split into training and validation subsets with keras.preprocessing module image_dataset_from_directory
- data augmentation, flipping and rotation of the images, was applied to reduce the risk of overfitting -although the model does show signs of overfitting in the end result
- the CNN structure :

      model = Sequential()
      model.add(Rescaling(1./255, input_shape=(150,150,3))) # Rescaling set up in the model itself !!!!!
      model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same'))
      model.add(LeakyReLU(alpha=0.1)) # prevents blocked and non-active RELUs
      model.add(MaxPooling2D((2, 2),padding='same'))
      model.add(Dropout(0.25)) # DROPOUT layer
      model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
      model.add(LeakyReLU(alpha=0.1))
      model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
      model.add(Dropout(0.25))  # DROPOUT layer
      model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
      model.add(LeakyReLU(alpha=0.1))
      model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
      model.add(Dropout(0.4)) # DROPOUT layer
      model.add(Flatten())
      model.add(Dense(128, activation='linear'))
      model.add(LeakyReLU(alpha=0.1))
      model.add(Dropout(0.3)) # DROPOUT layer
      model.add(Dense(num_classes, activation='sigmoid'))



