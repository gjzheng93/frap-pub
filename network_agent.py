import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
import random
from keras.engine.topology import Layer
import os
from keras.callbacks import EarlyStopping, TensorBoard
import pickle as pkl

from agent import Agent
import copy

rotation_matrix_2_p = np.zeros((4, 4))
j = 2
for i in range(4):
    rotation_matrix_2_p[i][j] = 1
    j = (j + 1) % 4

rotation_matrix_4_p = np.zeros((8, 8))
j = 4
for i in range(8):
    rotation_matrix_4_p[i][j] = 1
    j = (j + 1) % 8

class Selector(Layer):

    def __init__(self, select, d_phase_encoding, d_action, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.d_phase_encoding = d_phase_encoding
        self.d_action = d_action
        self.select_neuron = K.constant(value=self.select, shape=(1, self.d_phase_encoding))

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Selector, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        batch_size = K.shape(x)[0]
        constant = K.tile(self.select_neuron, (batch_size, 1))
        return K.min(K.cast(K.equal(x, constant), dtype="float32"), axis=-1, keepdims=True)

    def get_config(self):
        config = {"select": self.select, "d_phase_encoding": self.d_phase_encoding, "d_action": self.d_action}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.d_action)


def conv2d_bn(input_layer, index_layer,
              filters=16,
              kernel_size=(3, 3),
              strides=(1, 1)):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False,
                  name="conv{0}".format(index_layer))(input_layer)
    bn = BatchNormalization(axis=bn_axis, scale=False, name="bn{0}".format(index_layer))(conv)
    act = Activation('relu', name="act{0}".format(index_layer))(bn)
    pooling = MaxPooling2D(pool_size=2)(act)
    x = Dropout(0.3)(pooling)
    return x


class NetworkAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, best_round=None, bar_round=None):

        import tensorflow as tf
        import keras.backend.tensorflow_backend as KTF

        tf_config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(config=tf_config)
        KTF.set_session(session)

        super(NetworkAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path)

        # ===== check num actions == num phases ============

        #self.num_actions = self.dic_sumo_env_conf["ACTION_DIM"]
        #self.num_phases = self.dic_sumo_env_conf["NUM_PHASES"]
        if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
            self.num_actions = 2
        else:
            self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.num_phases = len(self.dic_traffic_env_conf["PHASE"])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))

        self.memory = self.build_memory()

        self.cnt_round = cnt_round

        if cnt_round == 0:
            # initialization
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.load_network("round_0")
            else:
                self.q_network = self.build_network()
            #self.load_network(self.dic_agent_conf["TRAFFIC_FILE"], file_path=self.dic_path["PATH_TO_PRETRAIN_MODEL"])
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                if best_round:
                    # use model pool
                    self.load_network("round_{0}".format(best_round))

                    if bar_round and bar_round != best_round and cnt_round > 10:
                        # load q_bar network from model pool
                        self.load_network_bar("round_{0}".format(bar_round))
                    else:
                        if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                            if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                                self.load_network_bar("round_{0}".format(
                                    max((best_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0)))
                            else:
                                self.load_network_bar("round_{0}".format(
                                    max(best_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0)))
                        else:
                            self.load_network_bar("round_{0}".format(
                                max(best_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0)))

                else:
                    # not use model pool
                    self.load_network("round_{0}".format(cnt_round-1))

                    if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                        if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                            self.load_network_bar("round_{0}".format(
                                max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0)))
                        else:
                            self.load_network_bar("round_{0}".format(
                                max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0)))
                    else:
                        self.load_network_bar("round_{0}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0)))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        # decay the epsilon

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

        decayed_lr = self.dic_agent_conf["LEARNING_RATE"] * pow(self.dic_agent_conf["LR_DECAY"], cnt_round)
        self.dic_agent_conf["LEARNING_RATE"] = max(decayed_lr, self.dic_agent_conf["MIN_LR"])

    @staticmethod
    def _unison_shuffled_copies(Xs, Y, sample_weight):
        p = np.random.permutation(len(Y))
        new_Xs = []
        for x in Xs:
            assert len(x) == len(Y)
            new_Xs.append(x[p])
        return new_Xs, Y[p], sample_weight[p]

    @staticmethod
    def _cnn_network_structure(img_features):
        conv1 = conv2d_bn(img_features, 1, filters=32, kernel_size=(8, 8), strides=(4, 4))
        conv2 = conv2d_bn(conv1, 2, filters=16, kernel_size=(4, 4), strides=(2, 2))
        img_flatten = Flatten()(conv2)
        return img_flatten

    @staticmethod
    def _shared_network_structure(state_features, dense_d):
        hidden_1 = Dense(dense_d, activation="relu", name="hidden_shared_1")(state_features)
        return hidden_1

    @staticmethod
    def _separate_network_structure(state_features, dense_d, num_actions, memo=""):
        hidden_1 = Dense(dense_d, activation="relu", name="hidden_separate_branch_{0}_1".format(memo))(state_features)
        hidden_2 = Dense(dense_d, activation="relu", name="hidden_separate_branch_{0}_2".format(memo))(hidden_1)
        q_values = Dense(num_actions, activation="linear", name="q_values_separate_branch_{0}".format(memo))(hidden_1)
        return q_values

    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network = load_model(os.path.join(file_path, "%s.h5" % file_name), custom_objects={"Selector": Selector})
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(os.path.join(file_path, "%s.h5" % file_name), custom_objects={"Selector": Selector})
        print("succeed in loading model %s"%file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def build_network(self):

        raise NotImplementedError

    def build_memory(self):

        return []

    def build_network_from_copy(self, network_copy):

        '''Initialize a Q network from a copy'''

        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"Selector": Selector})
        network.set_weights(network_weights)
        network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        return network


    def prepare_Xs_Y(self, memory, dic_exp_conf):

        # ==============
        # may need to seperate the normalization process
        # may need to make all the process in vector to speed up
        # ==============

        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory
        # forget
        else:
            if self.dic_agent_conf['PRIORITY']:
                print("priority")
                sample_slice = []
                num_sample_list = [
                                   int(self.dic_agent_conf["MAX_MEMORY_LEN"] * 1 / 4),
                                   int(self.dic_agent_conf["MAX_MEMORY_LEN"] * 1 / 4),
                                   int(self.dic_agent_conf["MAX_MEMORY_LEN"] * 1 / 4),
                                   int(self.dic_agent_conf["MAX_MEMORY_LEN"] * 1 / 4),
                                   ]
                for i in range(ind_end - 1, -1, -1):
                    one_sample = memory[i]
                    #print("one_sample", one_sample)
                    #ave_num_veh = int(np.average(np.array(one_sample[0]['lane_num_vehicle'])))
                    ave_num_veh = max(np.array(one_sample[0]['lane_num_vehicle']))
                    interval_id = ave_num_veh // 10
                    if num_sample_list[interval_id] > 0:
                        sample_slice.append(one_sample)
                        num_sample_list[interval_id] -= 1

                    if np.sum(np.array(num_sample_list)) == 0:
                        break
                print("end priority", num_sample_list)
                sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(sample_slice))
                sample_slice = random.sample(sample_slice, sample_size)
                ## log
                pkl.dump(sample_slice, file=open(os.path.join(self.dic_path['PATH_TO_WORK_DIRECTORY'], "train_round", "round_"+str(self.cnt_round), "update_sample.pkl"), "ab"))
                #f = open(os.path.join(self.dic_path['PATH_TO_WORK_DIRECTORY'], "train_round", "update_sample_log.txt"), "a")
                #f.write('%d, %d, %d, %d\n'%(num_sample_list[0], num_sample_list[1], num_sample_list[2], num_sample_list[3]))
            else:
                ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
                memory_after_forget = memory[ind_sta: ind_end]
                print("memory size after forget:", len(memory_after_forget))

                # sample the memory
                sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
                sample_slice = random.sample(memory_after_forget, sample_size)
                print("memory samples number:", sample_size)

            if self.dic_agent_conf["ROTATION_INPUT"]:

                new_sample_slice = []
                for sample in sample_slice:
                    #print(sample[0]['lane_num_vehicle'], sample[0]['cur_phase'])
                    if len(sample[0]['lane_num_vehicle']) == 8:
                        rotation_matrix = rotation_matrix_4_p
                    elif len(sample[0]['lane_num_vehicle']) == 4:
                        rotation_matrix = rotation_matrix_2_p

                    new_sample = copy.deepcopy(sample)
                    new_sample[0]['lane_num_vehicle'] = np.dot(new_sample[0]['lane_num_vehicle'],
                                                               rotation_matrix)
                    new_sample[0]['cur_phase'] = np.dot(new_sample[0]['cur_phase'],
                                                        rotation_matrix)
                    new_sample[2]['lane_num_vehicle'] = np.dot(new_sample[2]['lane_num_vehicle'],
                                                               rotation_matrix)
                    new_sample[2]['cur_phase'] = np.dot(new_sample[2]['cur_phase'],
                                                        rotation_matrix)
                    new_sample_slice.append(sample)
                    new_sample_slice.append(new_sample)
                sample_slice = new_sample_slice

        dic_state_feature_arrays = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        for i in range(len(sample_slice)):
            state, action, next_state, reward, instant_reward, _ = sample_slice[i]

            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                dic_state_feature_arrays[feature_name].append(state[feature_name])

            _state = []
            _next_state = []
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                _state.append([state[feature_name]])
                _next_state.append([next_state[feature_name]])
            target = self.q_network.predict(_state)

            next_state_qvalues = self.q_network_bar.predict(_next_state)

            if self.dic_agent_conf["LOSS_FUNCTION"] == "mean_squared_error":
                final_target = np.copy(target[0])
                final_target[action] = reward / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                       np.max(next_state_qvalues[0])
            elif self.dic_agent_conf["LOSS_FUNCTION"] == "categorical_crossentropy":
                raise NotImplementedError

            Y.append(final_target)

        self.Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                   self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
        self.Y = np.array(Y)


    def convert_state_to_input(self, s):
        
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            inputs = []
            if self.num_phases == 2:
                dic_phase_expansion = self.dic_traffic_env_conf["phase_expansion_4_lane"]
            else:
                dic_phase_expansion = self.dic_traffic_env_conf["phase_expansion"]
            for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature == "cur_phase":
                    inputs.append(np.array([dic_phase_expansion[s[feature][0]]]))
                else:
                    inputs.append(np.array([s[feature]]))
            return inputs
        else:
            return [np.array([s[feature]]) for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]


    def choose_action(self, count, state):

        ''' choose the best action for current state '''

        q_values = self.q_network.predict(self.convert_state_to_input(state))
        if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
            action = random.randrange(len(q_values[0]))
        else:  # exploitation
            action = np.argmax(q_values[0])

        return action

    def train_network(self, dic_exp_conf):

        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            epochs = 1000
        else:
            epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        if self.dic_agent_conf["EARLY_STOP_LOSS"] == "val_loss":

            early_stopping = EarlyStopping(
                monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

            hist = self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs,
                                      shuffle=False,
                                      verbose=2, validation_split=0.3, callbacks=[early_stopping])
        elif self.dic_agent_conf["EARLY_STOP_LOSS"] == "loss":
            early_stopping = EarlyStopping(
                monitor='loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

            hist = self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs,
                                      shuffle=False,
                                      verbose=2, callbacks=[early_stopping])
