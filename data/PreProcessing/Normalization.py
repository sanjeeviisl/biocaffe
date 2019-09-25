
from PIL import Image
import os

class Normalize():
    """
    Normalize several original images. Due to the source of the dataset, all images don't have the same size
    """
    def __init__(self, path_o, path_n):
        self.path_original = path_o
        self.path_to_save = path_n

    def maxSize(self, sizeMax):
        """
        Search the max size of the image (weight or height)
        :param sizeMax: size from the image
        :return:
        """
        if sizeMax[0] > sizeMax[1]:
            return sizeMax[0]
        else:
            return sizeMax[1]

    def dim_to_max(self, im):
        """
        Normalize the image if the the size are square
        :param im:
        :return:
        """
        background = Image.new('RGB', (512, 512), (255, 255, 255, 0))
        background.paste(im, (int((512 - im.size[0])/2), int((512 - im.size[1])/2)))
        return background

    def make_to_square(self, im):
        """
        Normalize the image aren't square
        :param im:
        :return:
        """
        size = (512, 512)
        maxS = self.maxSize(im.size)
        sub_background = Image.new('RGB', (maxS, maxS), (255, 255, 255, 0))
        sub_background.paste(im, (int((maxS - im.size[0])/2), int((maxS - im.size[1])/2)))
        sub_background.thumbnail(size, Image.ANTIALIAS)
        return sub_background

    def reduce(self):
        """
        Reduce the image size of 256*256 for Training 
        :return:
        """
        for dir in os.listdir(self.path_to_save):
            for img in os.listdir(self.path_to_save + dir +"/"):
                image_to_reduce = Image.open(self.path_to_save + dir + "/" + img)
                image_red = image_to_reduce.resize((256,256),  Image.ANTIALIAS)
                image_red.save(self.path_to_save + dir + "/" + img)

    def normalize_dimension_image(self):
        """
        Make all the images square
        :return:
        """
        for dir in os.listdir(self.path_original):
            for img in os.listdir(self.path_original + dir +"/"):
                image = Image.open(self.path_original + dir + "/" + img)
                if (image.size[0] == 512 and image.size[1] == 512 ):
                    f_img = image
                elif self.maxSize(image.size) < 512:
                    f_img = self.dim_to_max(image)
                else:
                    f_img = self.make_to_square(image)
                f_img.save(self.path_to_save + dir + "/" + img)
