import Training
import Make_Net
import os

class Main_Training():
    def __init__(self):
        self.make_proto = Make_Net.make_net()

    def create_directory_for_train(self, train_path):
        return train_path.split("/lmdb")[0]


    def get_Net_to_solve(self):
        """
        Create the Net and prototxt
        :return:
        """
        print("Net :LF_f")
        o_net ="LF_f"
        o_cpu_gpu = ""
        while True:
            o_cpu_gpu = str(input(" Train with GPU or CPU : "))
            if o_cpu_gpu == "GPU" or o_cpu_gpu == "CPU":
                print("Test Net with " + o_cpu_gpu)
                break
            else:
                print("Enter GPU or CPU")
        o_lmdb = ""
        while True:
            o_lmdb = str(input("Path to LMDB (path_to_lmdb/lmdb): "))
            if os.path.exists(o_lmdb) and "lmdb" in o_lmdb:
                break
            else:
                print("Your path is not correct")
        o_species = int(input("How many species are in your dataset? "))
        mean_path = " "
        while True:
            mean_path = input("Give the path to mean.npy: ")
            if os.path.exists(mean_path) and "mean.npy" in mean_path:
                break
            else:
                print("Your path is not correct")
        t_max_iter = str(input("Max Iteration : "))
        all_option = [o_net, o_cpu_gpu, o_lmdb, o_species]
        self.make_proto.set_option(all_option, mean_path, self.create_directory_for_train(o_lmdb), t_max_iter)
        self.make_proto.create_Net()

    def train(self):
        """
        Train Net
        :return:
        """
        path_for_training = ""
        while True:
            path_for_training = str(input("Path to solver and model : "))
            if os.path.exists(path_for_training):
                break
            else:
                print("Your path is not correct")
        p = path_for_training
        path_mapping = {"LF_f": (p + "/Net_Solver.prototxt")}
        t_option_net = "LF_f"
        t_opt_net = path_mapping[t_option_net]
        t_option_gcpu = ""
        while True:
            t_option_gcpu = input("Train with GPU or CPU : ")
            if t_option_gcpu == "GPU" or t_option_gcpu == "CPU":
                print("Test Net with " + t_option_gcpu)
                break
            else:
                print("Enter GPU or CPU")
        tr = Training.Training(t_option_gcpu, t_opt_net)
        tr.train()

    def get_started(self):
        """
        start Training
        :return:
        """
        print("Set Prototxt(1) or Train(2)")
        s_or_t = ""
        while True:
            if s_or_t == "1" or "2":
                s_or_t = str(input("1/2 : "))
                break
            else:
                print("Enter 1 or 2")
        if s_or_t == "1":
            self.get_Net_to_solve()
        elif s_or_t == "2":
            self.train()

