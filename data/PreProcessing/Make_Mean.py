
import subprocess
import caffe
import caffe.proto.caffe_pb2
import numpy as np

class Create_Mean():
    """
    Create the pixel mean of the train dataset.
    Call the mean creator from caffe
    """

    def make_mean(self, path, path_to_train_lmdb, path_to_caffe):
        mean_file = open(path + "mean.sh", "w")
        mean_file.write("#!/usr/bin/env sh\n")
        mean_file.write(path_to_caffe +"/compute_image_mean " + path_to_train_lmdb + "train_lmdb/ \\\n")
        mean_file.write("  "+path+"mean.binaryproto")
        mean_file.close()
        subprocess.call(["sh", path + "/mean.sh"])
        blob = caffe.proto.caffe_pb2.BlobProto()
        m = open(path+"mean.binaryproto", 'rb').read()
        blob.ParseFromString(m)
        arr_mean = np.array(caffe.io.blobproto_to_array(blob))
        out = arr_mean[0]
        np.save(path+'mean.npy', out)

