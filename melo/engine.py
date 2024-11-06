#===----------------------------------------------------------------------===#
#
# Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sophon.sail as sail
import time
import os
from typing import Union, Optional, List
import numpy as np
import logging


class Engine:

    def __init__(self, model_path="", device_id=0, graph_id=0, mode=sail.IOMode.DEVIO) :
        # 如果环境变量中没有设置device_id，则使用默认值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            print(">>>> device_id is in os.environ. and device_id = ",device_id)
        self.model_path = model_path
        self.device_id = device_id
        self.graph_id = graph_id
        self.mode = mode

        self.net = sail.Engine(model_path, device_id, mode)
        logging.info("load {} success!".format(model_path))
        self.graph_name = self.net.get_graph_names()[graph_id]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shapes = []
        self.output_shapes = []
        for i in range(len(self.input_names)):
            self.input_shapes.append(self.net.get_input_shape(self.graph_name, self.input_names[i]))
        for i in range(len(self.output_names)):
            self.output_shapes.append(self.net.get_output_shape(self.graph_name, self.output_names[i]))

    def __str__(self):
        return "Engine: model_path={}, device_id={}, graph_id={}, mode={}".format(self.model_path, self.device_id, self.graph_id, self.mode)


    def __call__(self, input_tensors:List, output_tensors:Optional[List[sail.Tensor]]=None, \
                    core_list:Optional[List[int]]=[]) -> Optional[List[np.ndarray]]:
        assert len(input_tensors) == len(self.input_names), "the number of input is not equal to the number of model input, {} vs {}".format(len(input_tensors), len(self.input_names))
        input_tensors_dict = {}
        input_shapes_dict = {}
        for i in range(len(self.input_names)):
            input_tensors_dict[self.input_names[i]] = input_tensors[i]
            if isinstance(input_tensors[i], np.ndarray):
                input_shapes_dict[self.input_names[i]] = list(input_tensors[i].shape)
            elif isinstance(input_tensors[i], sail.Tensor):
                input_shapes_dict[self.input_names[i]] = list(input_tensors[i].shape())
            else:
                raise TypeError("input ele {} type error, expect np.ndarray or sail.Tensor, but get {}".format(i, type(input_tensors[i])))
        input_tensors = input_tensors_dict
        input_shapes = input_shapes_dict
        
        if output_tensors is not None:
            assert len(output_tensors) == len(self.output_names), \
                "the number of output is not equal to the number of model output, {} vs {}".format(len(output_tensors), len(self.output_names))
            output_tensors_dict = {}
            for i in range(len(self.output_names)):
                output_tensors_dict[self.output_names[i]] = output_tensors[i]
            output_tensors = output_tensors_dict

        if isinstance(list(input_tensors.values())[0], np.ndarray):
            res = self.net.process(self.graph_name, input_tensors, core_list)
            res_list = []
            for i in range(len(self.output_names)):
                res_list.append(res[self.output_names[i]])
            return res_list
        elif isinstance(list(input_tensors.values())[0], sail.Tensor):
            assert output_tensors is not None, "output tensor is None!"
            self.net.process(self.graph_name, input_tensors, input_shapes, output_tensors, core_list)
