import os
import shutil
import sys
import Make_Id_Species
import make_species_files
import Normalization
import Data_Augmentation
from Build_LMDB import Create_Lmdb
from Make_train_val_file import Make_train_val_file
from Make_Mean import Create_Mean

class Main_Preprocessing():

    def __init__(self):
        """
        Create the Main_processing
        :return:
        """
        self.p = ""
        self.a = ""
        self.path_database = {}
        self.caffe_path = ""
        self.m_dir = make_species_files.Make_Directories_Species()
        self.lmdb_build = Create_Lmdb()


    def make_directories(self, opt):
        """
        Make directories for the preprocessing
        :param opt:
        :return:
        """
        dir_list = ['data', 'lmdb', "normalization", "augmentation"]
        for sub_directory in dir_list:
            os.makedirs(self.a + sub_directory)

    def create_data_base(self):
        """
        Create a path database
        :return:
        """
        path_v = ["data", "normalization", "augmentation", "lmdb"]
        for p in path_v:
            self.path_database[p] = self.a + p + "/"
        self.path_database["dataset"] = self.p


    def dataset_Leaf(self):
        """
        Create the data for the Foliage Dataset
        :return:
        """
        self.create_data_base()
        self.make_directories("other")
        print("Create id Text File ...")
        id_d = Make_Id_Species.Set_Id_Species(self.p +"train/", self.path_database["data"])
        id_d.set_list_id()
        id_d.set_id_dic()
        print("Create File ...")
        for directories_data in ["normalization", "augmentation"]:
            self.m_dir.make_directories(id_d.path_dict, self.path_database[directories_data])
        print("Normalize ...")
        norm_train = Normalization.Normalize(self.p + "train/", self.path_database["normalization"])
        norm_train.normalize_dimension_image()
        print("Augmentation ...")
        d_augm = Data_Augmentation.Data_Augmentation(self.path_database["normalization"], self.path_database["augmentation"])
        d_augm.create_augmentation(False)
        print("Normalize Validation...")
        shutil.rmtree(self.path_database["normalization"])
        os.makedirs(self.a + "normalization/")
        self.m_dir.make_directories(id_d.path_dict, self.path_database["normalization"])
        norm_test = Normalization.Normalize(self.p + "validation/", self.path_database["normalization"])
        norm_test.normalize_dimension_image()
        norm_test.reduce()
        m_file = Make_train_val_file()
        print("Create Text File ...")
        path_for_val_train={"train": "augmentation", "validation": "normalization"}
        for i in path_for_val_train:
            m_file.make_file(self.path_database[path_for_val_train[i]], self.path_database["data"]+i+".txt", id_d.path_dict)
        self.lmdb_build.set_lmdb(self.path_database, self.a, self.caffe_path+ "build/tools")
        mean = Create_Mean()
        mean.make_mean(self.a, self.a + "lmdb/", self.caffe_path+ "build/tools")

    def test_path(self, p):
        if p[-1] == "/":
            return p
        else:
            return p+"/"

    def get_startet(self):
        """
        Make data augmentation, normalize, and create the lmdb
        :return:
        """
        while True:
            #self.caffe_path = str(input("Path to Caffe : "))
            self.caffe_path = "/hdd/biocaffe/"
            if os.path.exists(self.caffe_path):
                self.caffe_path = self.test_path(self.caffe_path)
                break
            else:
                print("Your path is not correct")
        while True:
            #self.p = str(input("Path to Dataset: "))
            self.p = "/hdd/data/leaf_data_set/plantdisease/"
            if os.path.exists(self.p):
                self.p = self.test_path(self.p)
                break
            else:
                print("Your path is not correct")
        if not (os.path.exists(self.p + "train") and os.path.exists(self.p + "validation")):
            print("No validation or/and train directory found")
            sys.exit()
        while True:
            self.a = str(input("Path for Data: "))
	    self.a = "/hdd/plant_lmdb_data"
            if os.path.exists(self.a):
                self.a = self.test_path(self.a)
                break
            else:
                print("Your path is not correct")
        self.dataset_Leaf()
