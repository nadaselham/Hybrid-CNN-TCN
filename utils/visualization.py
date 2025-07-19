import matplotlib.pyplot as plt
import numpy as np

def overlay_keypoints(image, keypoints):
    for (x, y) in keypoints:
        plt.scatter(x, y, c='red', s=10)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
