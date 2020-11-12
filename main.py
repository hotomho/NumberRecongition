from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

mndata = MNIST('./data', gz=True)
images, labels = mndata.load_training()
for i in range(10):
    img = np.reshape(images[i], (28, 28))
    plt.imshow(img, cmap="Greys")
    plt.title(labels[i])
    plt.show()
