import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

img_files = np.sort(listdir("dino_dataset/"))
train_images = []
# train_labels = [0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,
#                 -0.99,-0.99,-0.99,-0.99,-0.99,-0.99,-0.99,-0.99,-0.99,-0.99,-0.99,-0.99,-0.99,-0.99,]

train_labels = [0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,
                0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,
                0, 0]
print(img_files)

for i in range(len(img_files)):
    im = np.array(cv2.imread("dino_dataset/" + img_files[i], 0)) / 255
    #plt.imshow(im)
    #plt.show()
    train_images.append(im)

train_images = np.asarray(train_images)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="sigmoid"),
    keras.layers.Dense(2, activation="softmax")
])
print(model.summary())
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=1000)

# p = model.predict(train_images)
# model.save('models/detect_obs.h5')

printscreen =  np.array(cv2.imread("dino_dataset/test_img.png", 0)) / 255
_printscreen = np.ones(shape=(120, 480))
for irow in range(2):
    for icol in range(8):
        _img = np.copy(printscreen[irow * 60: (irow * 60) + 59, icol * 60: (icol * 60) + 59])
        _res = np.expand_dims(cv2.resize(_img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC), axis=0)
        _predict = model.predict(_res)
        _result = np.argmax(_predict)
        print(_result)
        if _result < 1:
            _printscreen[irow * 60: (irow * 60) + 59, icol * 60: (icol * 60) + 59] = 0
        else:
            _printscreen[irow * 60: (irow * 60) + 59, icol * 60: (icol * 60) + 59] = 1