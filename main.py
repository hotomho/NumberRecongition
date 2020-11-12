from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

trainingData = MNIST(path='./data/training', gz=True)
trainingData_image, trainingData_labels = trainingData.load_training()
testingData = MNIST(path='./data/testing', gz=True)
testingData_image, testingData_labels = testingData.load_training()

# test:
for i in range(10):
    img = np.reshape(testingData_image[i], (28, 28))
    plt.imshow(img, cmap="Greys")
    plt.title(testingData_labels[i])
    plt.show()
