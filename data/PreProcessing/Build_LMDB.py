import subprocess

class Create_Lmdb():
    """
    Create the LMDB for validation and train file
    Write the sh file
    Call the LMDB creator from Caffe
    """
    def set_lmdb(self, path, path_to_data, path_to_caffe):
        """
        Create the LMDB, first create a sh file and then call the convert Tools from caffe
        :param path: path database
        :param path_to_data: path for save the txt files
        :param path_to_caffe: path to caffe
        :return: nothing
        """
        print("Create LMDB")
        path_to_image = [path['augmentation'], path['normalization']]
        path_to_data_txt = [path_to_data+'data/train.txt', path_to_data+'data/validation.txt']
        name_lmdb = [path_to_data + 'lmdb/train_lmdb', path_to_data +'lmdb/validation_lmdb']
        t_v = ["train_", "validation_"]
        for i in range(0, len(path_to_image)):
            lmdb_file = open(path_to_data + t_v[i]+ "lmdb.sh", "w")
            lmdb_file.write("#!/usr/bin/env sh\n")
            lmdb_file.write("GLOG_logtostderr=1 " +path_to_caffe+"/convert_imageset \\\n")
            lmdb_file.write("    --resize_height=0 \\\n")
            lmdb_file.write("    --resize_width=0 \\\n")
            lmdb_file.write("    --shuffle \\\n")
            lmdb_file.write("    "+path_to_image[i]+" \\\n")
            lmdb_file.write("    "+path_to_data_txt[i] + " \\\n")
            lmdb_file.write("    "+ name_lmdb[i])
            lmdb_file.close()
            subprocess.call(["sh", path_to_data + t_v[i]+ "lmdb.sh"])

