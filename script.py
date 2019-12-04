# collect the common function
# import script

# get traffic in one lane
def get_traffic_volume(traffic_file):
    # only support "cross" and "synthetic"
    if "cross" in traffic_file:
        sta = traffic_file.find("equal_") + len("equal_")
        end = traffic_file.find(".xml")
        return int(traffic_file[sta:end])

    elif "synthetic" in traffic_file:
        traffic_file_list = traffic_file.split("-")
        volume_list = []
        for i in range(2, 6):
            volume_list.append(int(traffic_file_list[i][2:]))

        vol = min(max(volume_list[0:2]), max(volume_list[2:]))

        return int(vol/100)*100
    elif "flow" in traffic_file:
        sta = traffic_file.find("flow_1_1_") + len("flow_1_1_")
        end = traffic_file.find(".json")
        return int(traffic_file[sta:end])

    elif "real" in traffic_file:
        sta = traffic_file.rfind("-") + 1
        end = traffic_file.rfind(".json")
        return int(traffic_file[sta:end])

    elif "hangzhou" in traffic_file:
        traffic = traffic_file.split(".json")[0]
        vol = int(traffic.split("_")[-1])
        return vol

## get total number of vehicles
## not very comprehensive

def get_total_traffic_volume(traffic_file):
    # only support "cross" and "synthetic"
    if "cross" in traffic_file:
        sta = traffic_file.find("equal_") + len("equal_")
        end = traffic_file.find(".xml")
        return int(traffic_file[sta:end]) * 4

    elif "synthetic" in traffic_file:
        sta = traffic_file.rfind("-") + 1
        end = traffic_file.rfind(".json")
        return int(traffic_file[sta:end])

    elif "flow" in traffic_file:
        sta = traffic_file.find("flow_1_1_") + len("flow_1_1_")
        end = traffic_file.find(".json")
        return int(traffic_file[sta:end]) * 4

    elif "real" in traffic_file:
        sta = traffic_file.rfind("-") + 1
        end = traffic_file.rfind(".json")
        return int(traffic_file[sta:end])

    elif "hangzhou" in traffic_file:
        traffic = traffic_file.split(".json")[0]
        vol = int(traffic.split("_")[-1])
        return vol
