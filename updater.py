import pickle
import os
from config import DIC_AGENTS
import shutil


class Updater:

    def __init__(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, best_round=None, bar_round=None):

        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_exp_conf = dic_exp_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.agent_name = self.dic_exp_conf["MODEL_NAME"]

        self.agent = DIC_AGENTS[self.agent_name](self.dic_agent_conf, self.dic_traffic_env_conf, self.dic_path, self.cnt_round)

    def load_sample(self):

        sample_set = []
        if self.dic_exp_conf["PRETRAIN"]:
            sample_file = open(os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"],
                                            "train_round", "total_samples" + ".pkl"), "rb")
        elif self.dic_exp_conf["AGGREGATE"]:
            sample_file = open(os.path.join(self.dic_path["PATH_TO_AGGREGATE_SAMPLES"],
                                            "aggregate_samples.pkl"), "rb")
        else:
            sample_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "total_samples" + ".pkl"), "rb")
        try:
            while True:
                sample_set += pickle.load(sample_file)
        except EOFError:
            pass

        self.agent.prepare_Xs_Y(sample_set, self.dic_exp_conf)

    def update_network(self):

        self.agent.train_network(self.dic_exp_conf)
        if self.dic_exp_conf["PRETRAIN"]:
            self.agent.q_network.save(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                    "%s.h5" % self.dic_exp_conf["TRAFFIC_FILE"][0]))
            shutil.copy(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],"%s.h5" % self.dic_exp_conf["TRAFFIC_FILE"][0]),
                        os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
        elif self.dic_exp_conf["AGGREGATE"]:
            self.agent.q_network.save("model/initial", "aggregate.h5")
            shutil.copy("model/initial/aggregate.h5",
                        os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
        else:
            self.agent.save_network("round_" + str(self.cnt_round))
