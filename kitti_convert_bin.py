import json
import os

import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
import pdb
from waymo_open_dataset import dataset_pb2 as open_dataset

__type_list = {'unknown':0, 'Vehicle':1, 'Pedestrian':2, 'Sign':3, 'Cyclist':4}

def _create_pd_file_example(path, json_data, objects):
    """Creates a prediction objects file."""
    kitti_file = open(path)
    for line in kitti_file.readlines():
        line = line.strip('\n').split()
        if line[0]=='unknown' or line[0]=='Sign':
            continue
        if line[15]=='0':
            continue
        o = metrics_pb2.Object()
        o.context_name = json_data["context_name"]
        o.frame_timestamp_micros = json_data["frame_timestamp_micros"]
        # if int(line[15]) > 5:
        #     o.difficulty = 1
        # else:
        #     o.difficulty = 2
        box = label_pb2.Label.Box()
        box.center_x = float(line[11])
        box.center_y = float(line[12])
        box.center_z = float(line[13])
        box.length = float(line[10])
        box.width = float(line[9])
        box.height = float(line[8])
        # box.length = float(line[8])
        # box.width = float(line[10])
        # box.height = float(line[9])
        box.heading = float(line[14])
        o.object.box.CopyFrom(box)
        # This must be within [0.0, 1.0]. It is better to filter those boxes with
        # small scores to speed up metrics computation.
        o.score = float(line[15])
        # if float(line[15])<1:
        #     o.score = float(line[15])
        # else:
        #     o.score = 0.5
        # if float(line[15])>=1:
        #     o.object.num_lidar_points_in_box = int(line[15])
        # For tracking, this must be set and it must be unique for each tracked
        # sequence.
        # o.object.id =
        # Use correct type.
        o.object.type = __type_list[line[0]]
        objects.objects.append(o)
    return objects


    # Add more objects. Note that a reasonable detector should limit its maximum
    # number of boxes predicted per frame. A reasonable value is around 400. A
    # huge number of boxes can slow down metrics computation.

    # Write objects to a file.

# objects = metrics_pb2.Objects()
# source_folder = "/data/yang_ye/Waymo/val/Label"
# json_file = "jsonFile.json"
# __type_list = {'unknown':0, 'Vehicle':1, 'Pedestrian':2, 'Sign':3, 'Cyclist':4}
#
# path = os.listdir(source_folder)
#
# with open(json_file, 'r') as load_f:
#     json_datas = json.load(load_f)
#     for i in range(len(path)):
#         print(i)
#         json_index = str('{0:06}'.format(i))+'.bin'
#         kitti_file_name = str('{0:06}'.format(i))+'.txt'
#         json_data = json_datas[json_index]
#         obj = _create_pd_file_example(os.path.join(source_folder, kitti_file_name), json_data)
#         if obj:
#             objects.objects.append(obj)
# f = open('eval.bin', 'wb')
# f.write(objects.SerializeToString())
# f.close()
