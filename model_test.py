import json
import os
import pickle
from config import DIC_AGENTS, DIC_ENVS
import numpy as np
import pandas as pd
from math import isnan


def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def downsample(path_to_log):
    path_to_pkl = os.path.join(path_to_log, "inter_0.pkl")
    with open(path_to_pkl, "rb") as f_logging_data:
        logging_data = pickle.load(f_logging_data)
    subset_data = logging_data[::10]
    os.remove(path_to_pkl)
    with open(path_to_pkl, "wb") as f_subset:
        pickle.dump(subset_data, f_subset)


def write_summary(dic_path, run_counts, cnt_round):

    record_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", "round_" + str(cnt_round))
    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", "test_results.csv")
    path_to_seg_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", "test_seg_results.csv")
    num_seg = run_counts//3600

    if cnt_round == 0:
        df_col = pd.DataFrame(columns=("round", "duration", "vec_in", "vec_out"))
        if num_seg > 1:
            list_seg_col = ["round"]
            for i in range(num_seg):
                list_seg_col.append("duration-" + str(i))
            df_seg_col = pd.DataFrame(columns=list_seg_col)
            df_seg_col.to_csv(path_to_seg_log, mode="a", index=False)
        df_col.to_csv(path_to_log, mode="a", index=False)

    # summary items (duration) from csv
    df_vehicle_inter_0 = pd.read_csv(os.path.join(record_dir, "vehicle_inter_0.csv"),
                                     sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                     names=["vehicle_id", "enter_time", "leave_time"])

    vehicle_in = sum([int(x) for x in (df_vehicle_inter_0["enter_time"].values > 0)])
    vehicle_out = sum([int(x) for x in (df_vehicle_inter_0["leave_time"].values > 0)])
    duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
    ave_duration = np.mean([time for time in duration if not isnan(time)])
    summary = {"round": [cnt_round], "duration": [ave_duration], "vec_in": [vehicle_in], "vec_out": [vehicle_out]}
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(path_to_log, mode="a", header=False, index=False)

    if num_seg > 1:
        list_duration_seg = [float('inf')] * num_seg
        nan_thres = 120
        for i, interval in enumerate(range(0, run_counts, 3600)):
            did = np.bitwise_and(df_vehicle_inter_0["enter_time"].values < interval + 3600,
                                 df_vehicle_inter_0["enter_time"].values > interval)
            duration_seg = df_vehicle_inter_0["leave_time"][did].values - df_vehicle_inter_0["enter_time"][
                did].values
            ave_duration_seg = np.mean([time for time in duration_seg if not isnan(time)])
            # print(traffic_file, round, i, ave_duration)
            real_traffic_vol_seg = 0
            nan_num_seg = 0
            for time in duration_seg:
                if not isnan(time):
                    real_traffic_vol_seg += 1
                else:
                    nan_num_seg += 1

            if nan_num_seg < nan_thres:
                list_duration_seg[i] = ave_duration_seg

        round_summary = {"round": [cnt_round]}
        for j in range(num_seg):
            key = "duration-" + str(j)
            if key not in round_summary.keys():
                round_summary[key] = [list_duration_seg[j]]
        round_summary = pd.DataFrame(round_summary)
        round_summary.to_csv(path_to_seg_log, mode="a", index=False, header=False)


def test(model_dir, cnt_round, run_cnt, dic_traffic_env_conf, if_gui):
    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d"%cnt_round
    dic_path = {}
    dic_path["PATH_TO_MODEL"] = model_dir
    dic_path["PATH_TO_WORK_DIRECTORY"] = records_dir

    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    with open(os.path.join(records_dir, "exp.conf"), "r") as f:
        dic_exp_conf = json.load(f)

    if os.path.exists(os.path.join(records_dir, "sumo_env.conf")):
        with open(os.path.join(records_dir, "sumo_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    elif os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)

    dic_exp_conf["RUN_COUNTS"] = run_cnt
    dic_traffic_env_conf["IF_GUI"] = if_gui

    # dump dic_exp_conf
    with open(os.path.join(records_dir, "test_exp.conf"), "w") as f:
        json.dump(dic_exp_conf, f)

    if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0  # dic_agent_conf["EPSILON"]  # + 0.1*cnt_gen
        dic_agent_conf["MIN_EPSILON"] = 0
    agent_name = dic_exp_conf["MODEL_NAME"]
    agent = DIC_AGENTS[agent_name](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path,
        cnt_round=1,  # useless
    )
    try:
        agent.load_network(model_round)

        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](path_to_log=path_to_log,
                                                               path_to_work_directory=dic_path[
                                                                   "PATH_TO_WORK_DIRECTORY"],
                                                               dic_traffic_env_conf=dic_traffic_env_conf)

        done = False
        state = env.reset()
        step_num = 0
        stop_cnt = 0
        while step_num < int(dic_exp_conf["TEST_RUN_COUNTS"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            for one_state in state:
                action = agent.choose_action(step_num, one_state)

                action_list.append(action)

            next_state, reward, done, _ = env.step(action_list)

            state = next_state
            step_num += 1
            stop_cnt += 1
        env.bulk_log()

        if dic_traffic_env_conf["DONE_ENABLE"]:
            run_cnt_log = open(os.path.join(records_dir, "test_stop_cnt_log.txt"), "a")
            run_cnt_log.write("%s, %10s, %d\n"%("test", "round_"+str(cnt_round), stop_cnt))
            run_cnt_log.close()

        write_summary(dic_path, run_cnt, cnt_round)
        #env.end_sumo()
        if not dic_exp_conf["DEBUG"]:
            path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
                                       model_round)
            # print("downsample", path_to_log)
            downsample(path_to_log)
            # print("end down")

    except:
        error_dir = model_round.replace("model", "errors")
        f = open(os.path.join(error_dir, "error_info.txt"), "a")
        f.write("round_%d fail to test model"%cnt_round)
        f.close()
        pass
        # import sys
        # sys.stderr.write("fail to test model_%"%model_round)
        # raise SystemExit(1)
