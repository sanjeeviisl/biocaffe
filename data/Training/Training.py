import subprocess
import caffe

class Training():
    def __init__(self, option, path_to_solver):
        self.cpu_gpu = option
        self.solv = path_to_solver

    def set_mode(self):
        if self.cpu_gpu == "cpu":
            caffe.set_mode_cpu()
        elif self.cpu_gpu == "gpu":
            caffe.set_mode_gpu()

    def train(self):
        caffe.set_mode_gpu()
        solver = caffe.get_solver(self.solv)
        solver.solve()
