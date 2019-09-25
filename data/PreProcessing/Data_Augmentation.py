from PIL import ImageEnhance
from PIL import Image

import os

class Data_Augmentation():

    def __init__(self, path_norm, path_augm):
        """
        Create the Data Augmentation
        :param path_norm: path to normalized images
        :param path_augm: path to save the augmented Images
        :return:
        """
        self.path_to_save = path_augm
        self.path = path_norm

    def make_new_name(self, name, new_string):
        """
        Create a new name for the augmented image
        :param name: old name
        :param new_string: string to add for the new name
        :return:
        """
        name_split = name.split(".")
        name_conc = name_split[0]+new_string
        new_name = name_conc + ".jpg"
        return new_name

    def zoom(self, image_for_zoom, size):
        """
        scale the images
        :param image_for_zoom: image
        :param size: how much the image will be scaled
        :return: transformed image
        """
        image_re = image_for_zoom.resize(size, Image.BICUBIC)
        if size[0] == 1024:
            image_z = image_re.crop((256, 256, 768, 768))
        else:
            image_z = image_re.crop((128, 128, 640, 640))
        return image_z

    def dezoom(self, image_for_dezoom, size):
        """
        Resize the scaled images
        :param image_for_dezoom: image
        :param size: size for normalized
        :return: transformed image
        """
        image_r = image_for_dezoom.resize(size, Image.ANTIALIAS)
        background = Image.new('RGB', (512, 512), (255, 255, 255, 0))
        background.paste(image_r, (int((512 - image_r.size[0])/2), int((512 - image_r.size[1])/2)))
        return background

    def reduce(self, image_to_reduce, size):
        """
        Reduce the size form the transformed images to 256,256
        :param image_to_reduce: image
        :param size: 256,256
        :return: transformed image
        """
        image_red = image_to_reduce.resize(size,  Image.ANTIALIAS)
        return image_red

    def bright(self, image_for_br, fac):
        """
        Change the Brightness
        :param image_for_br: image
        :param fac: factor for the transformation
        :return: transformed image
        """
        image_br = ImageEnhance.Brightness(image_for_br)
        return image_br.enhance(fac)


    def flip(self, image_for_flip, grad):
        """
        Rotate the image
        :param image_for_flip: Image
        :param grad: angle
        :return: transformed image
        """
        return image_for_flip.rotate(grad)

    def crop(self, image_for_crop, pos):
        """
        Crop for translation
        :param image_for_crop: Image
        :param pos: which coin to cropped
        :return:
        """
        if pos == 1:
            return image_for_crop.crop((0, 0, 384, 384))
        elif pos == 2:
            return image_for_crop.crop((0, 128, 384, 512))
        elif pos == 3:
            return image_for_crop.crop((128, 128, 512, 512))
        else:
            return image_for_crop.crop((128, 0, 512, 384))


    def augmentation(self, directory, image_name, file_pil_one, option, path_au):
        """

        :param directory: the directory
        :param image_name: string containing the name of the image
        :param file_pil_one: the original image
        :param option: Which augmentation will be created
        :param path_au: path to save
        :return:
        """
        size_norm = (256,256)
        original_image = file_pil_one
        path = path_au + directory + "/"
        grad = [90, 180, 270]
        image_lightness = self.bright(original_image, 1.5)
        if option == "flipping":
            name_flipped_original = ["_flip_o", "_flip_2_o", "_flip_3_o"]
            name_flipped_lightness = ["_flip", "_flip_2", "flip_3"]
            for g in range(0, len(grad)):
                flipped_image = self.flip(original_image, grad[g])
                flipped_image_red = self.reduce(flipped_image, size_norm)
                flipped_image_red.save(path + self.make_new_name(image_name, (name_flipped_original[g])))
                flipped_image_lightness = self.flip(image_lightness, grad[g])
                flipped_image_lightness_red = self.reduce(flipped_image_lightness, size_norm)
                flipped_image_lightness_red.save(path + self.make_new_name(image_name, (name_flipped_lightness[g])))
        elif option == "scaling":
            name_scaling = ["_flip_de_384", "_flip_1_de_384", "_flip_2_de_384"]
            file_zoom = self.zoom(original_image, (768, 768))
            file_normalized = self.dezoom(file_zoom, (384, 384))
            for g in range(0, len(grad)):
                if g == 0:
                    file_normalized_red = self.reduce(file_normalized, size_norm)
                    file_normalized_red.save(path + self.make_new_name(image_name, name_scaling[g]))
                elif g == 1:
                    file_scale_flipped = self.flip(file_normalized, grad[g])
                    file_scale_flipped_red = self.reduce(file_scale_flipped,size_norm)
                    file_scale_flipped_red.save(path + self.make_new_name(image_name, name_scaling[g]))
                else:
                    continue 
        elif option == "scaling_2":
            name_scaling_2 = ["_flip_de_128", "_flip_1_de_128", "_flip_2_de_128"]
            file_scaling_2 = self.dezoom(image_lightness, (256, 256))
            for g in range(0, len(grad)):
                if g == 1:
                    file_scaling_2_red = self.reduce(file_scaling_2, size_norm)
                    file_scaling_2_red.save(path + self.make_new_name(image_name, name_scaling_2[g]))
                elif g == 0:
                    file_scale_2_flipped = self.flip(file_scaling_2, grad[g])
                    file_scale_2_flipped_red = self.reduce(file_scale_2_flipped, size_norm)
                    file_scale_2_flipped_red.save(path + self.make_new_name(image_name, name_scaling_2[g]))
                else:
                    continue
        elif option == "lightness":
            light = [1.5, 1.8]
            name_light = ["_br8", "br5"]
            for l in range(0, len(light)):
                file_lightness = self.bright(original_image, light[l])
                file_lightness_red = self.reduce(file_lightness, size_norm)
                file_lightness_red.save(path + self.make_new_name(image_name, name_light[l]))
        elif option == "translate":
            cropping = [1, 2, 3, 4]
            cropping_name = [[], [], [], []]
            for el in cropping_name:
                for x in range(0, 4):
                    el.append("")
            for i in range(0, 4):
                for j in range(0, 4):
                    if j == 0:
                        cropping_name[i][j] = ("_crop" + str(i+1))
                    else:
                        cropping_name[i][j] = ("_crop" + str(i+1) + "_flip_" + str(j+1))
            for c in range(0, len(cropping)):
                file_cropped = self.crop(original_image, cropping[c])
                file_cropped_red = self.reduce(file_cropped, size_norm)
                file_cropped_red.save(path + self.make_new_name(image_name, cropping_name[c][0]))
                for g in range(0, len(grad)):
                    file_flip_for_crop = self.flip(original_image, grad[g])
                    file_cropped = self.crop(file_flip_for_crop, cropping[c])
                    file_cropped_red = self.reduce(file_cropped, size_norm)
                    file_cropped_red.save(path + self.make_new_name(image_name, cropping_name[c][g+1]))


    def add_original_data(self):
        """
        Add the original image to the augmentation data
        :return:
        """
        for d in os.listdir(self.path):
            for f in os.listdir(self.path +"/" + d):
                ori = Image.open(self.path + "/" + d + "/" + f)
                red_ori = self.reduce(ori, (256, 256))
                red_ori.save(self.path_to_save + "/" + d + "/" + f)


    def create_augmentation(self, LF):
        """
        Create the augmentation
        :return:
        """
        transformation = ["flipping", "scaling", "scaling_2", "lightness", "translate"]
        for d in os.listdir(self.path):
            for f in os.listdir(self.path +"/" + d):
                for t in transformation:
                    image = Image.open(self.path + d + "/" + f)
                    self.augmentation(d, f, image, t, self.path_to_save)
        self.add_original_data()
