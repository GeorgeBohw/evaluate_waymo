# Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for waymo_open_dataset.metrics.python.detection_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import pdb

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2

ERROR = 1e-6


class DetectionMetricsEstimatorTest(tf.test.TestCase):

    def get_boxes_from_bin(self, file):
        pd_bbox, pd_type, pd_frame_id, pd_score, difficulty = [], [], [], [], []
        stuff1 = metrics_pb2.Objects()
        with open(file, 'rb')as rf:
            stuff1.ParseFromString(rf.read())
            for i in range(len(stuff1.objects)):
                obj = stuff1.objects[i].object
                pd_frame_id.append(stuff1.objects[i].frame_timestamp_micros)
                box = [obj.box.center_x, obj.box.center_y, obj.box.center_z,
                       obj.box.length, obj.box.width, obj.box.height, obj.box.heading]
                pd_bbox.append(box)
                pd_score.append(stuff1.objects[i].score)
                pd_type.append(obj.type)

                if obj.num_lidar_points_in_box and obj.num_lidar_points_in_box<=5:
                    difficulty.append(2)
                else:
                    difficulty.append(1)
        return np.array(pd_bbox), np.array(pd_type), np.array(pd_frame_id), np.array(pd_score), np.array(difficulty)

    def get_boxes_from_txt(self, file_path):
        __type_list = {'unknown': 0, 'Vehicle': 1, 'Pedestrian': 2, 'Sign': 3, 'Cyclist': 4}
        pd_bbox, pd_type, pd_frame_id, pd_score, difficulty = [], [], [], [], []
        for i in range(39987):
            file_name = str('{0:06}'.format(i)) + '.txt'
            file = os.path.join(file_path, file_name)
            if not os.path.exists(file):
                continue
            with open(file, 'r')as f:
                for line in f.readlines():
                    line = line.strip('\n').split()
                    if float(line[15])==0:
                        continue
                    pd_frame_id.append(int(file_name.split('.')[0]))
                    box = [float(line[11]), float(line[12]), float(line[13]),
                           float(line[10]), float(line[9]), float(line[8]),float(line[14])]
                    pd_bbox.append(box)
                    if float(line[15])<1: # pd
                        pd_score.append(line[15])
                    else: # gt
                        pd_score.append(0.5)
                    pd_type.append(__type_list[line[0]])
                    if float(line[15])>5:
                        difficulty.append(1)
                    else:
                        difficulty.append(2)
        return np.array(pd_bbox), np.array(pd_type), np.array(pd_frame_id), np.array(pd_score), np.array(difficulty)

    def _BuildConfig(self):
        config = metrics_pb2.Config()
        # pdb.set_trace()
        config_text = """
    num_desired_score_cutoffs: 11
    breakdown_generator_ids: OBJECT_TYPE
    breakdown_generator_ids: RANGE
    difficulties {
    levels: 1
    levels: 2
    }
    difficulties {
    levels: 1
    levels: 2
    }
    matcher_type: TYPE_HUNGARIAN
    iou_thresholds: 0.5
    iou_thresholds: 0.7
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    box_type: TYPE_3D
    """
        text_format.Merge(config_text, config)
        return config

    def _BuildGraph(self, graph):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._gt_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
            metrics = detection_metrics.get_detection_metric_ops(
                config=self._BuildConfig(),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=tf.zeros_like(
                    self._pd_frame_id, dtype=tf.bool),
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                # ground_truth_difficulty=tf.ones_like(self._gt_frame_id, dtype=tf.uint8),
                ground_truth_difficulty=self._gt_difficulty,
                recall_at_precision=0.95,
            )
            return metrics

    def _EvalUpdateOps(
            self,
            sess,
            graph,
            metrics,
            prediction_frame_id,
            prediction_bbox,
            prediction_type,
            prediction_score,
            ground_truth_frame_id,
            ground_truth_bbox,
            ground_truth_type,
            ground_truth_difficulty,
    ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
                self._gt_difficulty: ground_truth_difficulty,
            })

    def _EvalValueOps(self, sess, graph, metrics):
        ddd = {}
        for item in metrics.items():
            ddd[item[0]] = sess.run([item[1][0]])
        return ddd

    def testAPBasic(self):
        print("start")
        print(pd_file)
        # pd_bbox, pd_type, pd_frame_id, pd_score, _ = self.get_boxes_from_txt(pd_file)
        gt_bbox, gt_type, gt_frame_id, _, difficulty = self.get_boxes_from_bin(gt_file)
        print("111111111111111111111111111111111111111111111111111111")
        # gt_bbox, gt_type, gt_frame_id, _, difficulty = self.get_boxes_from_txt(gt_file)
        pd_bbox, pd_type, pd_frame_id, pd_score, _ = self.get_boxes_from_bin(pd_file)
        print("111111111111111111111111111111111111111111111111111111")

        # pd_bbox, pd_type, pd_frame_id, pd_score, difficulty = self.get_boxes_from_bin(pd_file)
        # gt_bbox, gt_type, gt_frame_id = pd_bbox, pd_type, pd_frame_id
        graph = tf.Graph()
        metrics = self._BuildGraph(graph)
        with self.test_session(graph=graph) as sess:
            sess.run(tf.compat.v1.initializers.local_variables())
            self._EvalUpdateOps(sess, graph, metrics, pd_frame_id, pd_bbox, pd_type,
                                pd_score, gt_frame_id, gt_bbox, gt_type, difficulty)

            aps = self._EvalValueOps(sess, graph, metrics)
            for key, value in aps.items():
                print(key, ":", value)
            print("111111111111111111111111111111111111111111111111111111")

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    pd_file = 'bin/eval.bin'
    gt_file = 'bin/gt.bin'
    # pd_file = '/home/yang_ye/SA-SSD_Waymo/tools/result_val'
    # gt_file = '/data/yang_ye/Waymo/val/Label'
    tf.compat.v1.disable_eager_execution()
    tf.test.main()
