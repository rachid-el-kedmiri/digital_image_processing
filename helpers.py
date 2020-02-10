from matplotlib  import pyplot as plt
from collections.abc import Iterable
import cv2   as cv
import numpy as np


def show_image_with_histogram(images, show_cdf=False):
    """"
    Display images by side their histogram

    Parameters
    ----------
    image : array of images or a single image
    embed_cdf: whether to include cumulative distributive 
              function in the histogram plot default false
    """

    if not isinstance(images, Iterable):
        images = [images, ]

    image_count, subplot_index = len(images), 1

    plt.figure(
        num=None, 
        figsize=(16, 3*image_count), 
        dpi=100, 
        facecolor='w', 
        edgecolor='k'
    )

    for image in images:
        histogram, _ = np.histogram(image.flatten(),256,[0,256])
        # histogram = cv.calcHist([image],[0],None,[256],[0,256])
        plt.subplot(image_count, 2, subplot_index)
        plt.imshow(image, 'gray')

        plt.subplot(image_count, 2, subplot_index + 1)
        plt.hist(image.flatten(),256,[0,256], color = 'r')

        if show_cdf:
            cdf = histogram.cumsum()
            cdf_normalized = cdf * histogram.max()/ cdf.max()
            plt.plot(cdf_normalized, color = 'b')
            plt.legend(('cdf','histogram'), loc = 'upper left')

        subplot_index += 2

    plt.xlim([0,256])
    plt.show()
