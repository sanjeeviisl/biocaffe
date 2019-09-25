import os

class Create_Solver():
    """
    Create the Solver for final Net
    """

    def __init__(self, t_d, max_iteration):
        self.train_directory = t_d
        self.max_iter = max_iteration

    def solver(self, option):
        cpu_gpu = option
        param = ["test_iter","test_interval","base_lr","lr_policy","gamma","stepsize","display","max_iter", "weight_decay","momentum","snapshot","snapshot_prefix","solver_mode"]
        value = ["210", "2500", "0.001", "\"step\"", "0.1", "100000", "200", self.max_iter, "0.0002", "0.9", "2500", "\""+str(self.train_directory+"/snapshot/") +"\"", cpu_gpu]
        s = open(str(self.train_directory + "/Net_Solver.prototxt"), 'w')
        s.write("net:\"" +str(self.train_directory + "/Net.prototxt") +"\"\n")
        for p in range(0, len(param)):
            s.write(param[p] + ": " + value[p] +"\n")
        s.close()
