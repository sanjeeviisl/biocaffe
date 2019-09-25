from cnn.Net_final import Create_Proto as LF_f, Create_Solver as Solver_LF_f
import numpy as np
import os

class make_net():
    def __init__(self):
        self.option = []

    def set_option(self, opt, m_path, path_to_train, iter):
        """
        Create the Net prototxt
        :param opt: which Net version
        :param m_path: the path to train mean
        :return:
        """
        for o in opt:
            self.option.append(o)
            self.mean_path = m_path
            self.path_t = path_to_train
            self.m_iter = iter

    def create_Net(self):
        if not os.path.exists(self.path_t + "/snapshot/"):
            os.makedirs(self.path_t + "/snapshot/")
        mean_all = np.load(self.mean_path)
        mean_all = mean_all.mean(1).mean(1)
        mean_rgb = []
        for m in mean_all:
            mean_rgb.append(int(m))
        proto = LF_f.Create_Proto(int(self.option[3]), self.option[2], mean_rgb, self.path_t)
        proto.set_solver()
        solver = Solver_LF_f.Create_Solver(self.path_t, self.m_iter)
        solver.solver(self.option[1])


