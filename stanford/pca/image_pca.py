# -- coding: utf-8
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class ImagePca:
    def __init__(self, n_component, image_path):
        self.components = n_component
        self.image_path = image_path

    def load_image(self):
        im = Image.open(self.image_path).convert("L")
        width, height = im.size
        print "width:%s, height:%s" % (width, height)
        data = (np.array(im.getdata(), dtype=float) / 255.0).reshape((height, width))
        return data


def main():
    image_pca = ImagePca(50, "D:\\kaggle\\pca\\scenery.jpg")
    original_image = image_pca.load_image()
    pca = PCA(50)
    new_image = pca.fit_transform(original_image)
    plt.figure("new")
    plt.imshow(pca.inverse_transform(new_image))
    plt.show()
if __name__ == '__main__':
    main()