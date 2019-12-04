import numpy as np
import os
import sys
import pickle
from sys import platform
from math import floor, ceil
import pandas as pd
import json

# ================
# initialization checed
# need to check get state
# ================


if platform == "linux" or platform == "linux2":
    # this is linux
    try:
        os.environ['SUMO_HOME'] = '/usr/share/sumo'
        sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
        import traci
        import traci.constants as tc
    except ImportError:
        try:
            os.environ['SUMO_HOME'] = '/headless/sumo'
            import traci
            import traci.constants as tc
        except ImportError:
            if "SUMO_HOME" in os.environ:
                print(os.path.join(os.environ["SUMO_HOME"], "tools"))
                sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
                import traci
                import traci.constants as tc
            else:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

elif platform == "win32":
    os.environ['SUMO_HOME'] = 'D:\\software\\sumo-0.32.0'

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            import traci
            import traci.constants as tc
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

elif platform =='darwin':
    os.environ['SUMO_HOME'] = "/Users/{0}/sumo/".format(os.getlogin())

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            import traci
            import traci.constants as tc
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

else:
    sys.exit("platform error")




def get_traci_constant_mapping(constant_str):
    return getattr(tc, constant_str)

class Intersection:

    def __init__(self, light_id, list_vehicle_variables_to_sub, dic_sumo_env_conf):
        '''
        still need to automate generation
        '''

        self.node_light = "inter{0}".format(light_id)
        self.list_vehicle_variables_to_sub = list_vehicle_variables_to_sub

        # ===== sumo intersection settings =====

        self.list_approachs = [str(i) for i in range(dic_sumo_env_conf["N_LEG"])]
        self.dic_approach_to_node = {str(i): "{0}.node{1}".format(self.node_light, i) for i in self.list_approachs }
        self.dic_entering_approach_to_edge = {approach: "edge-{0}-{1}".format(self.dic_approach_to_node[approach], self.node_light) for approach in self.list_approachs}
        self.dic_exiting_approach_to_edge = {approach: "edge-{0}-{1}".format(self.node_light, self.dic_approach_to_node[approach]) for approach in self.list_approachs}

        self.lane_direc = []
        self.lane_direc += ["r" for i in range(dic_sumo_env_conf["LANE_NUM"]["RIGHT"])]
        self.lane_direc += ["t" for i in range(dic_sumo_env_conf["LANE_NUM"]["STRAIGHT"])]
        self.lane_direc += ["l" for i in range(dic_sumo_env_conf["LANE_NUM"]["LEFT"])]
        self.num_lane = len(self.lane_direc)
        self.l_lane_ind = [i for i in range(self.num_lane)]


        self.dic_entering_approach_lanes = {str(i): self.l_lane_ind for i in self.list_approachs}
        self.dic_exiting_approach_lanes = {str(i): self.l_lane_ind for i in self.list_approachs}

        # grid settings
        self.length_lane = 300
        self.length_terminal = 50
        self.length_grid = 5
        self.num_grid = int(self.length_lane//self.length_grid)

        # generate all lanes
        self.list_entering_lanes = []
        for approach in self.list_approachs:
            self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach]+'_'+str(i) for i in self.dic_entering_approach_lanes[approach]]
        self.list_exiting_lanes = []
        for approach in self.list_approachs:
            self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + '_' + str(i) for i in self.dic_exiting_approach_lanes[approach]]
        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        # generate signals
        self.list_phases = dic_sumo_env_conf["PHASE"]
        self.dic_app_offset = {str(i): int(i) for i in self.list_approachs}

        self.dic_phase_strs = {}

        for p in self.list_phases:
            list_default_str = ["r" for i in range(self.num_lane*len(self.list_approachs))]

            # set green for right turn
            for any_app in self.list_approachs:
                for ind_this_direc in np.where(np.array(self.lane_direc) == "r")[0].tolist():
                    list_default_str[self.dic_app_offset[any_app] * self.num_lane + ind_this_direc] = 'g'

            app1 = p[0]
            direc1 = p[1]
            app2 = p[3]
            direc2 = p[4]

            for ind_this_direc in np.where(np.array(self.lane_direc) == direc1.lower())[0].tolist():
                list_default_str[self.dic_app_offset[app1] * self.num_lane + ind_this_direc] = 'G'
            for ind_this_direc in np.where(np.array(self.lane_direc) == direc2.lower())[0].tolist():
                list_default_str[self.dic_app_offset[app2] * self.num_lane + ind_this_direc] = 'G'
            self.dic_phase_strs[p] = "".join(list_default_str)

        self.all_yellow_phase_str = "".join(["y" for i in range(self.num_lane*len(self.list_approachs))])
        self.all_red_phase_str = "".join(["r" for i in range(self.num_lane*len(self.list_approachs))])

        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        # initialization

        # -1: all yellow, -2: all red, -3: none
        self.current_phase_index = 0
        self.previous_phase_index = 0
        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

        self.dic_lane_sub_current_step = None
        self.dic_lane_sub_previous_step = None
        self.dic_vehicle_sub_current_step = None
        self.dic_vehicle_sub_previous_step = None
        self.list_vehicles_current_step = []
        self.list_vehicles_previous_step = []

        self.dic_vehicle_min_speed = {}  # this second
        self.dic_vehicle_arrive_leave_time = dict() # cumulative

        self.dic_feature = {} # this second

    def set_signal(self, action, action_pattern, yellow_time, all_red_time):

        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time: # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                traci.trafficlights.setRedYellowGreenState(
                    self.node_light, self.dic_phase_strs[self.list_phases[self.current_phase_index]])
                self.all_yellow_flag = False
            else:
                pass
        else:
            # determine phase
            if action_pattern == "switch": # switch by order
                if action == 0: # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1: # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases)
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set": # set to certain phase
                self.next_phase_to_set_index = action

            # set phase
            if self.current_phase_index == self.next_phase_to_set_index: # the light phase keeps unchanged
                pass
            else: # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                traci.trafficlights.setRedYellowGreenState(
                    self.node_light, self.all_yellow_phase_str)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    def update_previous_measurements(self):

        self.previous_phase_index = self.current_phase_index
        self.dic_lane_sub_previous_step = self.dic_lane_sub_current_step
        self.dic_vehicle_sub_previous_step = self.dic_vehicle_sub_current_step
        self.list_vehicles_previous_step = self.list_vehicles_current_step

    def update_current_measurements(self):
        ## need change, debug in seeing format

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        # ====== lane level observations =======

        self.dic_lane_sub_current_step = {lane: traci.lane.getSubscriptionResults(lane) for lane in self.list_lanes}

        # ====== vehicle level observations =======

        # get vehicle list
        self.list_vehicles_current_step = traci.vehicle.getIDList()
        list_vehicles_new_arrive = list(set(self.list_vehicles_current_step) - set(self.list_vehicles_previous_step))
        list_vehicles_new_left = list(set(self.list_vehicles_previous_step) - set(self.list_vehicles_current_step))
        list_vehicles_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicles_new_left_entering_lane = []
        for l in list_vehicles_new_left_entering_lane_by_lane:
            list_vehicles_new_left_entering_lane += l

        # update subscriptions
        for vehicle in list_vehicles_new_arrive:
            traci.vehicle.subscribe(vehicle, [getattr(tc, var) for var in self.list_vehicle_variables_to_sub])

        # vehicle level observations
        self.dic_vehicle_sub_current_step = {vehicle: traci.vehicle.getSubscriptionResults(vehicle) for vehicle in self.list_vehicles_current_step}

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicles_new_arrive)
        self._update_left_time(list_vehicles_new_left_entering_lane)

        # update vehicle minimum speed in history
        self._update_vehicle_min_speed()

        # update feature
        self._update_feature()

    # ================= update current step measurements ======================

    def _update_leave_entering_approach_vehicle(self):

        list_entering_lane_vehicle_left = []

        # update vehicles leaving entering lane
        if self.dic_lane_sub_previous_step is None:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append(
                    list(
                        set(self.dic_lane_sub_previous_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]) - \
                        set(self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")])
                    )
                )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicles_arrive):

        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicles_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                print("vehicle already exists!")
                sys.exit(-1)

    def _update_left_time(self, list_vehicles_left):

        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicles_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_vehicle_min_speed(self):
        '''
        record the minimum speed of one vehicle so far
        :return:
        '''
        dic_result = {}
        for vec_id, vec_var in self.dic_vehicle_sub_current_step.items():
            speed = vec_var[get_traci_constant_mapping("VAR_SPEED")]
            if vec_id in self.dic_vehicle_min_speed: # this vehicle appeared in previous time stamps:
                dic_result[vec_id] = min(speed, self.dic_vehicle_min_speed[vec_id])
            else:
                dic_result[vec_id] = speed
        self.dic_vehicle_min_speed = dic_result

    def _update_feature(self):

        dic_feature = dict()

        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["vehicle_position_img"] = None #self._get_lane_vehicle_position(self.list_entering_lanes)
        dic_feature["vehicle_speed_img"] = None #self._get_lane_vehicle_speed(self.list_entering_lanes)
        dic_feature["vehicle_acceleration_img"] = None
        dic_feature["vehicle_waiting_time_img"] = None #self._get_lane_vehicle_accumulated_waiting_time(self.list_entering_lanes)

        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres01"] = self._get_lane_num_vehicle_been_stopped(0.1, self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres1"] = self._get_lane_num_vehicle_been_stopped(1, self.list_entering_lanes)
        dic_feature["lane_queue_length"] = self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_left"] = None
        dic_feature["lane_sum_duration_vehicle_left"] = None
        dic_feature["lane_sum_waiting_time"] = self._get_lane_sum_waiting_time(self.list_entering_lanes)

        dic_feature["terminal"] = None

        self.dic_feature = dic_feature

    # ================= calculate features from current observations ======================

    def _get_lane_queue_length(self, list_lanes):
        '''
        queue length for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_HALTING_NUMBER")]
                for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_NUMBER")]
                for lane in list_lanes]

    def _get_lane_sum_waiting_time(self, list_lanes):
        '''
        waiting time for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("VAR_WAITING_TIME")]
                for lane in list_lanes]

    def _get_lane_list_vehicle_left(self, list_lanes):
        '''
        get list of vehicles left at each lane
        ####### need to check
        '''

        return None

    def _get_lane_num_vehicle_left(self, list_lanes):

        list_lane_vehicle_left = self._get_lane_list_vehicle_left(list_lanes)
        list_lane_num_vehicle_left = [len(lane_vehicle_left) for lane_vehicle_left in list_lane_vehicle_left]
        return list_lane_num_vehicle_left

    def _get_lane_sum_duration_vehicle_left(self, list_lanes):

        ## not implemented error
        raise NotImplementedError

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):

        list_num_of_vec_ever_stopped = []
        for lane in list_lanes:
            cnt_vec = 0
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                if self.dic_vehicle_min_speed[vec] < thres:
                    cnt_vec += 1
            list_num_of_vec_ever_stopped.append(cnt_vec)

        return list_num_of_vec_ever_stopped

    def _get_position_grid_along_lane(self, vec):
        pos = int(self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_LANEPOSITION")])
        return min(pos//self.length_grid, self.num_grid)


    def _get_lane_vehicle_position(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.zeros(self.num_grid)
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = 1
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    def _get_lane_vehicle_speed(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.full(self.num_grid, fill_value=np.nan)
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_SPEED")]
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    def _get_lane_vehicle_accumulated_waiting_time(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.full(self.num_grid, fill_value=np.nan)
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_ACCUMULATED_WAITING_TIME")]
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    # ================= get functions from outside ======================

    def get_current_time(self):
        return traci.simulation.getCurrentTime() / 1000

    def get_dic_vehicle_arrive_leave_time(self):

        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):

        return self.dic_feature

    def get_state(self, list_state_features):

        dic_state = {state_feature_name: self.dic_feature[state_feature_name] for state_feature_name in list_state_features}

        return dic_state

    def get_reward(self, dic_reward_info):

        dic_reward = dict()
        dic_reward["flickering"] = None
        dic_reward["sum_lane_queue_length"] = None
        dic_reward["sum_lane_wait_time"] = None
        dic_reward["sum_lane_num_vehicle_left"] = None
        dic_reward["sum_duration_vehicle_left"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres01"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(self.dic_feature["lane_num_vehicle_been_stopped_thres1"])

        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward

    def _get_vehicle_info(self, veh_id):
        try:
            pos = self.dic_vehicle_sub_current_step[veh_id][get_traci_constant_mapping("VAR_LANEPOSITION")]
            speed = self.dic_vehicle_sub_current_step[veh_id][get_traci_constant_mapping("VAR_SPEED")]
            return pos, speed
        except:
            return None, None



class SumoEnv:

    # add more variables here if you need more measurements
    LIST_LANE_VARIABLES_TO_SUB = [
        "LAST_STEP_VEHICLE_NUMBER",
        "LAST_STEP_VEHICLE_ID_LIST",
        "LAST_STEP_VEHICLE_HALTING_NUMBER",
        "VAR_WAITING_TIME",

    ]

    # add more variables here if you need more measurements
    LIST_VEHICLE_VARIABLES_TO_SUB = [
        "VAR_POSITION",
        "VAR_SPEED",
        # "VAR_ACCELERATION",
        # "POSITION_LON_LAT",
        "VAR_WAITING_TIME",
        "VAR_ACCUMULATED_WAITING_TIME",
        # "VAR_LANEPOSITION_LAT",
        "VAR_LANEPOSITION",
    ]

    def _get_sumo_cmd(self):

        if platform == "linux" or platform == "linux2":
            if os.environ['SUMO_HOME'] == '/usr/share/sumo':
                sumo_binary = r"/usr/bin/sumo-gui"
                sumo_binary_nogui = r"/usr/bin/sumo"
            elif os.environ['SUMO_HOME'] == '/headless/sumo':
                sumo_binary = r"/headless/sumo/bin/sumo-gui"
                sumo_binary_nogui = r"/headless/sumo/bin/sumo"
            else:
                sys.exit("linux sumo binary path error")
            # for FIB-Server
            #sumo_binary = r"/usr/bin/sumo/bin/sumo-gui"
            #sumo_binary_nogui = r"/usr/bin/sumo"
        elif platform == "darwin":
            sumo_binary = r"/opt/local/bin/sumo-gui"
            sumo_binary_nogui = r"/opt/local/bin/sumo"
        elif platform == "win32":
            sumo_binary = r'D:\\software\\sumo-0.32.0\\bin\\sumo-gui.exe'
            sumo_binary_nogui = r'D:\\software\\sumo-0.32.0\\bin\\sumo.exe'
        else:
            sys.exit("platform error")

        real_path_to_sumo_files = os.path.join(os.path.split(os.path.realpath(__file__))[0], self.path_to_work_directory, "cross.sumocfg")

        sumo_cmd = [sumo_binary,
                   '-c',
                   r'{0}'.format(real_path_to_sumo_files),
                   "--step-length",
                   str(self.dic_traffic_env_conf["INTERVAL"])
                    ]

        sumo_cmd_nogui = [sumo_binary_nogui,
                         '-c',
                         r'{0}'.format(real_path_to_sumo_files),
                         "--step-length",
                         str(self.dic_traffic_env_conf["INTERVAL"])
                          ]

        if self.dic_traffic_env_conf["IF_GUI"]:
            return sumo_cmd
        else:
            return sumo_cmd_nogui

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):

        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.sumo_cmd_str = self._get_sumo_cmd()

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print ("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            #raise ValueError

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):

        # initialize intersections
        # self.list_intersection = [Intersection(i, self.LIST_VEHICLE_VARIABLES_TO_SUB) for i in range(self.dic_sumo_env_conf["NUM_INTERSECTIONS"])]

        self.list_intersection = []
        for i in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            for j in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
                self.list_intersection.append(Intersection("{0}_{1}".format(i, j), self.LIST_VEHICLE_VARIABLES_TO_SUB, self.dic_traffic_env_conf))

        self.list_inter_log = [[] for i in range(len(self.list_intersection))]
        # get lanes list
        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        print ("start sumo")
        while True:
            try:
                traci.start(self.sumo_cmd_str)
                break
            except:
                continue
        print ("succeed in start sumo")

        # start subscription
        for lane in self.list_lanes:
            traci.lane.subscribe(lane, [getattr(tc, var) for var in self.LIST_LANE_VARIABLES_TO_SUB])

        # get new measurements
        for inter in self.list_intersection:
            inter.update_current_measurements()

        state, done = self.get_state()

        return state

    @staticmethod
    def convert_dic_to_df(dic):
        list_df = []
        for key in dic:
            df = pd.Series(dic[key], name=key)
            list_df.append(df)
        return pd.DataFrame(list_df)

    def bulk_log(self):

        valid_flag = {}
        for inter_ind in range(len(self.list_intersection)):
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = self.convert_dic_to_df(dic_vehicle)
            df.to_csv(path_to_log_file, na_rep="nan")

            inter = self.list_intersection[inter_ind]
            feature = inter.get_feature()
            print(feature['lane_num_vehicle'])
            if max(feature['lane_num_vehicle']) > self.dic_traffic_env_conf["VALID_THRESHOLD"]:
                valid_flag[inter_ind] = 0
            else:
                valid_flag[inter_ind] = 1
        json.dump(valid_flag, open(os.path.join(self.path_to_log, "valid_flag.json"), "w"))

        for inter_ind in range(len(self.list_inter_log)):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()


    def end_sumo(self):
        traci.close()

    def get_current_time(self):
        return traci.simulation.getCurrentTime() / 1000

    def get_feature(self):

        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):

        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
        done = self._check_episode_done(list_state)

        return list_state, done

    def get_reward(self):

        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in self.list_intersection]

        return list_reward

    # def log(self, cur_time, before_action_feature, action):
    #
    #     for inter_ind in range(len(self.list_intersection)):
    #         path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
    #         f = open(path_to_log_file, "ab+")
    #         pickle.dump(
    #             {"time": cur_time,
    #              "state": before_action_feature[inter_ind],
    #              "action": action[inter_ind]}, f)
    #         f.close()

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                    "state": before_action_feature[inter_ind],
                                                    "action": action[inter_ind]})

    def step(self, action):

        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action = 0
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()

            before_action_feature = self.get_feature()
            state = self.get_state()

            if self.dic_traffic_env_conf["DEBUG"]:
                print("time: {0}, phase: {1}, time this phase: {2}, action: {3}".format(instant_time, before_action_feature[0]["cur_phase"], before_action_feature[0]["time_this_phase"], action_in_sec_display[0]))
            else:
                if i == 0:
                    print("time: {0}, phase: {1}, time this phase: {2}, action: {3}".format(instant_time,
                                                                                        before_action_feature[0][
                                                                                            "cur_phase"],
                                                                                        before_action_feature[0][
                                                                                            "time_this_phase"],
                                                                                        action_in_sec_display[0]))

            # _step
            self._inner_step(action_in_sec)

            # get reward
            reward = self.get_reward()
            average_reward_action = (average_reward_action*i + reward[0])/(i+1)

            # log
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)

            next_state, done = self.get_state()

        return next_state, reward, done, [average_reward_action]

    def _inner_step(self, action):

        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # set signals
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_conf["ALL_RED_TIME"]
            )

        # run one step

        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            traci.simulationStep()

        # get new measurements
        for inter in self.list_intersection:
            inter.update_current_measurements()

        #self.log_lane_vehicle_position()
        if self.dic_traffic_env_conf["LOG_DEBUG"]:
            self.log_first_vehicle()
            self.log_phase()

    def _check_episode_done(self, list_state):

        # ======== to implement ========

        return False

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str

        dic_lane_map = {
            "edge1-0_0": "w",
            "edge2-0_0": "e",
            "edge3-0_0": "s",
            "edge4-0_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " + list_to_str(inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt" % dic_lane_map[lane]),
                                "a"))

    def log_first_vehicle(self):
        _veh_id = "1."
        _veh_id_2 = "3."
        _veh_id_3 = "4."
        _veh_id_4 = "6."
        for inter in self.list_intersection:
            for i in range(100):
                veh_id = _veh_id + str(i)
                veh_id_2 = _veh_id_2 + str(i)
                pos, speed = inter._get_vehicle_info(veh_id)
                pos_2, speed_2 = inter._get_vehicle_info(veh_id_2)
                #print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_a")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_a"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_b")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_b"))

                if pos and speed:
                    print("%f, %f, %f" % (self.get_current_time(), pos, speed),
                          file=open(os.path.join(self.path_to_log, "first_vehicle_info_a", "first_vehicle_info_a_%d.txt"%i), "a"))
                if pos_2 and speed_2:
                    print("%f, %f, %f" % (self.get_current_time(), pos_2, speed_2),
                          file=open(os.path.join(self.path_to_log, "first_vehicle_info_b", "first_vehicle_info_b_%d.txt"%i), "a"))

                veh_id_3 = _veh_id_3 + str(i)
                veh_id_4 = _veh_id_4 + str(i)
                pos_3, speed_3 = inter._get_vehicle_info(veh_id_3)
                pos_4, speed_4 = inter._get_vehicle_info(veh_id_4)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_c")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_c"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_d")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_d"))

                if pos_3 and speed_3:
                    print("%f, %f, %f" % (self.get_current_time(), pos_3, speed_3),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_c", "first_vehicle_info_a_%d.txt" % i),
                              "a"))
                if pos_4 and speed_4:
                    print("%f, %f, %f" % (self.get_current_time(), pos_4, speed_4),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_d", "first_vehicle_info_b_%d.txt" % i),
                              "a"))

        #for inter in self.list_intersection:
        #    pos, speed = inter._get_vehicle_info(veh_id)
        #    pos_2, speed_2 = inter._get_vehicle_info(veh_id_2)
        #    if pos and speed:
        #        print("%f, %f, %f" % (self.get_current_time(), pos, speed),
        #              file=open(os.path.join(self.path_to_log, "first_vehicle_info.txt"), "a"))
        #    if pos_2 and speed_2:
        #        print("%f, %f, %f" % (self.get_current_time(), pos_2, speed_2),
        #              file=open(os.path.join(self.path_to_log, "first_vehicle_info_2.txt"), "a"))

    def log_phase(self):
        for inter in self.list_intersection:
            print("%f, %f" % (self.get_current_time(), inter.current_phase_index),
                  file=open(os.path.join(self.path_to_log, "log_phase.txt"), "a"))

