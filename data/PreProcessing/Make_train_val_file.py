import os

class Make_train_val_file():
    #Create the train.txt and validation.txt file to build the lmdb

    def make_file(self, path_to_directoy, path_to_data, id_list):
        """
        create the .txt file
        :param path_to_directoy : the directory where images for train or validation are save
        :param path_to_data: path to save the the .txt file
        :param id_list: the id list which map the species name to a label
        :return:
        """
        training = open(path_to_data, "w")
        for dir in os.listdir(path_to_directoy):
            for file in os.listdir(path_to_directoy + dir + "/"):
                training.write(dir + "/" + file + " " + str(id_list[dir]) +"\n")
        training.close()

