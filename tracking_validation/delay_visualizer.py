from rosbag_parser import get_topics_dict
from matplotlib import pyplot as plt
import numpy as np


topic_legends_map = {
    "/perception/object_recognition/objects": "prediction",
    "/perception/object_recognition/tracking/objects": "tracking",
    "/perception/object_recognition/detection/objects": "detection",
}


def plot_time_delay(t_dicts:dict, msg:str):
    pair_list = t_dicts[msg]
    t_diff = []
    for stamp, msg in pair_list:
        time:float = stamp * 1e-9
        orig_time:float = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        t_diff.append([orig_time, time - orig_time])
    np_t_diff = np.array(t_diff).reshape(-1, 2)
    plt.plot(np_t_diff[:,0], np_t_diff[:,1],".")



def main(bag_file:str, topic_list:list):
    t_dicts = get_topics_dict(bag_file, topic_list)
    plt.figure()
    legends = []
    for msg in topic_list:
        plot_time_delay(t_dicts, msg)
        legends.append(topic_legends_map[msg])
    plt.legend(legends)
    plt.grid()
    plt.show()





if __name__ == "__main__":
    rosbag_file_path = "/home/yoshiri/autoware_bag/prediction_data/cutindata/202211_cutin_data/no18_ego30k_tag20k_20m_vy0.5ms_b64370f4-8b54-4cdd-863d-11d942f6bc7f_2022-11-10-11-42-57/b64370f4-8b54-4cdd-863d-11d942f6bc7f_2022-11-10-11-42-57_0.db3"
    rosbag_file_path = "/home/yoshiri/autoware_bag/SPkadai/193/193-2/048d965a-a74e-41bf-96d3-4834a45ed1a1/211130ae-8a64-4311-95ab-1b8406c4499b_2023-09-13-11-42-41_p0900_3.db3"
    rosbag_file_path = "/home/yoshiri/autoware_bag/SPkadai/178/0388f3fc-7d88-4520-b78f-b90befa92094_2023-08-29-10-33-16_2.db3"
    rosbag_file_path = "/home/yoshiri/autoware_bag/SPkadai/144/211130ae-8a64-4311-95ab-1b8406c4499b_2023-07-18-11-42-38_p0900_5.db3"
    rosbag_file_path = "/home/yoshiri/autoware_bag/SPkadai/124/01d060d9-8d25-45e0-b45e-9fd7201ac27b_2023-05-26-11-30-03_p0900_3.db3"
    rosbag_file_path = "/home/yoshiri/autoware_bag/SPkadai/104/211130ae-8a64-4311-95ab-1b8406c4499b_2022-12-21-11-59-36_p0900_9.db3"

    topic_list = ["/perception/object_recognition/objects", "/perception/object_recognition/tracking/objects", "/perception/object_recognition/detection/objects"]

    main(rosbag_file_path, topic_list)