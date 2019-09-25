import os

class Make_Directories_Species():
    """
    Create directories from the species name
    """

    def make_directories(self, id_list, path):
        for species in id_list:
            os.makedirs(path + species)
