from caffe import layers, params
import caffe


class Create_Proto():
    """
    Create the final Net
    """
    def __init__(self, number_of_species, o, mean, t_d):
        self.path_to_lmdb = self.set_path_lmdb(o)
        self.species = number_of_species
        self.mean = mean
        self.train_directory = t_d

    def set_path_lmdb(self, option):
        lmdb_path = []
        p = option
        p_to_lmdb_test = p + ("/validation_lmdb/")
        lmdb_path.append(p_to_lmdb_test)
        p_to_lmdb_train = p + ("/train_lmdb/")
        lmdb_path.append(p_to_lmdb_train)
        return lmdb_path

    def set_proto(self, opt_dep):
        net = caffe.NetSpec()
        if opt_dep:
            net.data = layers.Input(input_param=dict(shape=dict(dim=[1, 3, 256, 256])))
        else:
            net.data, net.label = layers.Data(batch_size=10, backend=params.Data.LMDB, source=self.path_to_lmdb[0], include=dict(phase=1),
                             transform_param=dict(mirror=False, mean_value=self.mean), ntop=2)
        net.conv1a = layers.Convolution(net.data, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], kernel_size=9, num_output=32, pad=4, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu1a = layers.ReLU(net.conv1a, in_place=True)
        net.conv1b = layers.Convolution(net.conv1a, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], kernel_size=9, num_output=32, pad=4, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu1b = layers.ReLU(net.conv1b, in_place=True)
        net.pool1 = layers.Pooling(net.conv1b, pooling_param= dict(pool=params.Pooling.MAX, kernel_size=2, stride=2))
        net.conv2a = layers.Convolution(net.pool1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], kernel_size=5, num_output=64, pad=2, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu2a = layers.ReLU(net.conv2a, in_place=True)
        net.conv2b = layers.Convolution(net.conv2a, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], kernel_size=5, num_output=64, pad=2, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu2b = layers.ReLU(net.conv2b, in_place=True)
        net.pool2 = layers.Pooling(net.conv2b, pooling_param= dict(pool=params.Pooling.MAX, kernel_size=2, stride=2))
        net.conv3a = layers.Convolution(net.pool2, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], kernel_size=3, num_output=128, pad=1, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu3a = layers.ReLU(net.conv3a, in_place=True)
        net.conv3b = layers.Convolution(net.conv3a, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], kernel_size=3, num_output=128, pad=1, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu3b = layers.ReLU(net.conv3b, in_place=True)
        net.pool3 = layers.Pooling(net.conv3b, pooling_param= dict(pool=params.Pooling.MAX, kernel_size=2, stride=2))
        net.conv4a = layers.Convolution(net.pool3, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], kernel_size=3, num_output=256, pad=1, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu4a = layers.ReLU(net.conv4a, in_place=True)
        net.conv4b = layers.Convolution(net.conv4a, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], kernel_size=3, num_output=256, pad=1, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu4b = layers.ReLU(net.conv4b, in_place=True)
        net.pool4 = layers.Pooling(net.conv4b, pooling_param= dict(pool=params.Pooling.MAX, kernel_size=2, stride=2))
        net.conv5a = layers.Convolution(net.pool4, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], kernel_size=3, num_output=512, pad=1, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu5a = layers.ReLU(net.conv5a, in_place=True)
        net.conv5b = layers.Convolution(net.conv5a, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], kernel_size=3, num_output=512, pad=1, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu5b = layers.ReLU(net.conv5b, in_place=True)
        net.pool5 = layers.Pooling(net.conv5b, pooling_param= dict(pool=params.Pooling.MAX, kernel_size=2, stride=2))
        net.conv6 = layers.Convolution(net.pool5, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], kernel_size=3, num_output=768, pad=1, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu6 = layers.ReLU(net.conv6, in_place=True)
        net.pool6 = layers.Pooling(net.conv6, pooling_param= dict(pool=params.Pooling.MAX, kernel_size=2, stride=2))
        net.ip1 = layers.InnerProduct(net.pool6, num_output=2048, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu7 = layers.ReLU(net.ip1, in_place=True)
        net.drop1 = layers.Dropout(net.ip1, in_place=True, dropout_param=dict(dropout_ratio=0.4))
        net.ip2 = layers.InnerProduct(net.ip1, num_output=2048, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        net.relu8 = layers.ReLU(net.ip2, in_place=True)
        net.drop2 = layers.Dropout(net.ip2, in_place=True, dropout_param=dict(dropout_ratio=0.4))
        net.ip3 = layers.InnerProduct(net.ip2, num_output=self.species, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0))
        if not opt_dep:
            net.accuracy = layers.Accuracy(net.ip3, net.label, include=dict(phase=1), accuracy_param=dict(top_k=1))
            if self.species>=5:
                net.accuracy_5 = layers.Accuracy(net.ip3, net.label, include=dict(phase=1), accuracy_param=dict(top_k=5))
            net.loss = layers.SoftmaxWithLoss(net.ip3, net.label)
        else:
            net.prob = layers.Softmax(net.ip3)
        return net.to_proto()

    def set_proto_train(self):
            net = caffe.NetSpec()
            net.data, net.label = layers.Data(batch_size=10, backend=params.Data.LMDB, source=self.path_to_lmdb[1], include=dict(phase=0),
                         transform_param=dict(mirror=False, mean_value=self.mean), ntop=2)
            return net.to_proto()

    def set_solver(self):
        with open(str(self.train_directory + "/Net.prototxt"), 'w') as f:
            f.write('%s\n' % str(self.set_proto_train()))
            f.write('%s\n'% str(self.set_proto(False)))
        with open(str(self.train_directory + "/lf.prototxt"), 'w') as d:
            d.write('%s\n'% str(self.set_proto(True)))
