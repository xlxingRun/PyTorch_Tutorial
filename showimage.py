import matplotlib.pyplot as plt
import numpy as np
import torchvision
from dataset import classes, train_loader


# functions to show an image
def imgShow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def Show():
    # get some random training images
    data_iter = iter(train_loader)
    images, labels = data_iter.next()

    # show images
    imgShow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
