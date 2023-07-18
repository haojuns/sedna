# Copyright 2021 The KubeEdge Authors.
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

"""
Divide multiple tasks based on data

Parameters
----------
samples： Train data, see `sedna.datasources.BaseDataSource` for more detail.

Returns
-------
tasks: All tasks based on training data.
task_extractor: Model with a method to predicting target tasks
"""

import time

from sedna.common.file_ops import FileOps
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.seen_task_learning.artifact import Task

__all__ = ('UpdateStrategyDefault', )


@ClassFactory.register(ClassType.KM)
class UpdateStrategyDefault:
    """
    Decide processing strategies for different tasks

    Parameters
    ----------
    task_index: str or Dict
    """

    def __init__(self, task_index, **kwargs):
        if isinstance(task_index, str):
            task_index = FileOps.load(task_index)
        self.task_index = task_index

    def __call__(self, samples, task_type):
        """
        Parameters
        ----------
        samples: BaseDataSource
            seen task samples or unseen task samples to be processed.
        task_type: str
            "seen_task" or "unseen_task".
            See sedna.common.constant.KBResourceConstant for more details.

        Returns
        -------
        self.tasks: List[Task]
            tasks to be processed.
        task_update_strategies: Dict
            strategies to process each task.
        """

        if task_type == "seen_task":
            task_index = self.task_index["seen_task"]# self.task_index["seen_task"]:  {'task_groups': [<sedna.algorithms.seen_task_learning.artifact.TaskGroup object at 0x7f3d717f0c40>], 'extractor': './workspace/benchmarkingjob/rfnet_lifelong_learning/927bdf3a-142e-11ee-ba17-a906087290a8/output/train/1/seen_task/seen_task_task_attr_extractor.pkl'}
        else:
            task_index = self.task_index["unseen_task"]#  self.task_index["unseen_task"]:  {'task_groups': [], 'extractor': './workspace/benchmarkingjob/rfnet_lifelong_learning/927bdf3a-142e-11ee-ba17-a906087290a8/output/train/1/unseen_task/unseen_task_task_attr_extractor.pkl'}
        #注意！！！！！！！！！！！！！！！！！！！！！，unseentask没有task_groups
        self.extractor = task_index["extractor"]
        task_groups = task_index["task_groups"]

        tasks = [task_group.tasks[0] for task_group in task_groups]

        task_update_strategies = {}
        for task in tasks:
            task_update_strategies[task.entry] = {
                "raw_data_update": None,
                "target_model_update": None,
                "task_attr_update": None,
            }
        #task_update_strategies:   {'semantic_segamentation_model': {'raw_data_update': None, 'target_model_update': None, 'task_attr_update': None}}
        x_data = samples.x
        y_data = samples.y
        d_type = samples.data_type
        
        
        
        
        # 对samples再次进行task allocation：假设两个任务，任务一和二会被分别分配对应的样本的x/y，task_update_strategies的值不同任务间相等（task.entry当前都是'semantic_segamentation_model'）
        '''for task in tasks:
            origin = task.meta_attr
            _x = [x for x in x_data if origin in x[0]]    #x_data: array([['','','',...]],  x_data[0]: array(['/data/user8302433/fc/dataset/curb-detection/train_data/images/real_aachen_000019_000019_leftImg8bit.png'],dtype='<U103'),   x_data[0][0]:   '/data/user8302433/fc/dataset/curb-detection/train_data/images/real_aachen_000019_000019_leftImg8bit.png'
            _y = [y for y in y_data if origin in y]

            task_df = BaseDataSource(data_type=d_type)
            task_df.x = _x
            task_df.y = _y

            task.samples = task_df

            task_update_strategies[task.entry] = {
                "raw_data_update": samples,
                "target_model_update": samples,
                "task_attr_update": samples
            }'''
            
        for task in tasks:
            origin = task.meta_attr

            task_df = BaseDataSource(data_type=d_type)
            task_df.x = x_data
            task_df.y = y_data

            task.samples = task_df

            task_update_strategies[task.entry] = {
                "raw_data_update": samples,
                "target_model_update": samples,
                "task_attr_update": samples
            }
        #task_update_strategies:  {'semantic_segamentation_model': {'raw_data_update': <sedna.datasources.BaseDataSource object at 0x7f3d717f0af0>, 'target_model_update': <sedna.datasources.BaseDataSource object at 0x7f3d717f0af0>, 'task_attr_update': <sedna.datasources.BaseDataSource object at 0x7f3d717f0af0>}}
        #保存每一个任务的不同策略对应的样本？
        return tasks, task_update_strategies
