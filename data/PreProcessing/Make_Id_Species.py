
import os

class Set_Id_Species():

    def __init__(self, path_to_dataset, path_to_data):
        """
        Create the id label to the species name
        :param path_to_dataset: path to the dataset
        :param path_to_data: path to save the id.txt file
        :return:
        """
        self.path_to_id = path_to_data
        self.path_to_original_data = path_to_dataset
        self.path_dict = {}

    def set_list_id(self):
        """
        from the dataset directory, create the id.txt to map the species name to the label
        :return:
        """
        id_species = open(self.path_to_id + "id.txt", "w")
        counter_species = 0
        for directories in os.listdir(self.path_to_original_data):
            id_species.write(directories + " " + str(counter_species) + '\n')
            counter_species += 1
        id_species.close()

    def set_id_dic(self):
        """
        make a directories for the species to label
        :return:
        """
        id_data = open(self.path_to_id + "id.txt")
        for line in id_data:
            name, label = line.split(" ")
            self.path_dict[name] = int(label)
        id_data.close()
