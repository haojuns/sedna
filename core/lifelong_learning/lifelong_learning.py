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
import time

import numpy as np

from sedna.core.base import JobBase
from sedna.common.file_ops import FileOps
from sedna.common.constant import K8sResourceKind, K8sResourceKindStatus
from sedna.common.constant import KBResourceConstant
from sedna.common.config import Context
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.seen_task_learning.seen_task_learning import SeenTaskLearning
from sedna.algorithms.unseen_task_processing import UnseenTaskProcessing
from sedna.service.client import KBClient
from sedna.algorithms.knowledge_management.cloud_knowledge_management \
    import CloudKnowledgeManagement
from sedna.algorithms.knowledge_management.edge_knowledge_management \
    import EdgeKnowledgeManagement


class LifelongLearning(JobBase):
    """
     Lifelong Learning (LL) is an advanced machine learning (ML) paradigm that
     learns continuously, accumulates the knowledge learned in the past, and
     uses/adapts it to help future learning and problem solving.

     Sedna provide the related interfaces for application development.

    Parameters
    ----------
    estimator : Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.
    task_definition : Dict
        Divide multiple tasks based on data,
        see `task_definition.task_definition` for more detail.
    task_relationship_discovery : Dict
        Discover relationships between all tasks, see
        `task_relationship_discovery.task_relationship_discovery` for more detail.
    task_allocation : Dict
        Mining seen tasks of inference sample,
        see `task_allocation.task_allocation` for more detail.
    task_remodeling : Dict
        Remodeling tasks based on their relationships,
        see `task_remodeling.task_remodeling` for more detail.
    inference_integrate : Dict
        Integrate the inference results of all related
        tasks, see `inference_integrate.inference_integrate` for more detail.
    task_update_decision: Dict
        Task update strategy making algorithms,
        see 'knowledge_management.task_update_decision.task_update_decision' for more detail.
    unseen_task_allocation: Dict
        Mining unseen tasks of inference sample,
        see `unseen_task_processing.unseen_task_allocation.unseen_task_allocation` for more detail.
    unseen_sample_recognition: Dict
        Dividing inference samples into seen tasks and unseen tasks,
        see 'unseen_task_processing.unseen_sample_recognition.unseen_sample_recognition' for more detail.
    unseen_sample_re_recognition: Dict
        Dividing unseen training samples into seen tasks and unseen tasks,
        see 'unseen_task_processing.unseen_sample_re_recognition.unseen_sample_re_recognition' for more detail.

    Examples
    --------
    >>> estimator = XGBClassifier(objective="binary:logistic")
    >>> task_definition = {
            "method": "TaskDefinitionByDataAttr",
            "param": {"attribute": ["season", "city"]}
        }
    >>> task_relationship_discovery = {
            "method": "DefaultTaskRelationDiscover", "param": {}
        }
    >>> task_mining = {
            "method": "TaskMiningByDataAttr",
            "param": {"attribute": ["season", "city"]}
        }
    >>> task_remodeling = None
    >>> inference_integrate = {
            "method": "DefaultInferenceIntegrate", "param": {}
        }
    >>> task_update_decision = {
            "method": "UpdateStrategyDefault", "param": {}
        }
    >>> unseen_task_allocation = {
            "method": "UnseenTaskAllocationDefault", "param": {}
        }
    >>> unseen_sample_recognition = {
            "method": "SampleRegonitionDefault", "param": {}
        }
    >>> unseen_sample_re_recognition = {
            "method": "SampleReRegonitionDefault", "param": {}
        }
    >>> ll_jobs = LifelongLearning(
            estimator,
            task_definition=None,
            task_relationship_discovery=None,
            task_allocation=None,
            task_remodeling=None,
            inference_integrate=None,
            task_update_decision=None,
            unseen_task_allocation=None,
            unseen_sample_recognition=None,
            unseen_sample_re_recognition=None,
        )
    """

    def __init__(self,
                 estimator,
                 task_definition=None,
                 task_relationship_discovery=None,
                 task_allocation=None,
                 task_remodeling=None,
                 inference_integrate=None,
                 task_update_decision=None,
                 unseen_task_allocation=None,
                 unseen_sample_recognition=None,
                 unseen_sample_re_recognition=None
                 ):     
        
        self.seen_task_learning = SeenTaskLearning(
            estimator=estimator,
            task_definition=task_definition,
            task_relationship_discovery=task_relationship_discovery,
            seen_task_allocation=task_allocation,
            task_remodeling=task_remodeling,
            inference_integrate=inference_integrate
        )
        
        import pdb
        #pdb.set_trace()
        
        '''pdb_dict={'self': <sedna.core.lifelong_learning.lifelong_learning.LifelongLearning object at 0x7fdc7ae43fa0>, 
         'estimator': <basemodel.BaseModel object at 0x7fdd756bfa60>, 
         'task_definition': <task_definition_all.TaskDefinitionAll object at 0x7fdd756bf820>, 
         'task_relationship_discovery': None, 
         'task_allocation': <task_allocation_all.TaskAllocationAll object at 0x7fdd51f3cb50>, 
         'task_remodeling': None, 
         'inference_integrate': None, 
         'task_update_decision': <task_update_decision_all.TaskUpdateDecisionAll object at 0x7fdc466ce460>, 
         'unseen_task_allocation': None, 
         'unseen_sample_recognition': None, 
         'unseen_sample_re_recognition': None, 
         'pdb': <module 'pdb' from '/data/user8302433/anaconda3/envs/new_ianvs/lib/python3.8/pdb.py'>, 
         '__class__': <class 'sedna.core.lifelong_learning.lifelong_learning.LifelongLearning'>}'''

        self.unseen_sample_recognition = unseen_sample_recognition or {
            "method": "SampleRegonitionDefault"
        }
        self.unseen_sample_recognition_param = self.seen_task_learning._parse_param(
            self.unseen_sample_recognition.get("param", {}))

        self.unseen_sample_re_recognition = unseen_sample_re_recognition or {
            "method": "SampleReRegonitionDefault"
        }
        
        self.unseen_sample_re_recognition_param = self.seen_task_learning._parse_param(
            self.unseen_sample_re_recognition.get("param", {}))

        #self.task_update_decision = task_update_decision or {
            #"method": "UpdateStrategyDefault"
        #}
        
        self.task_update_decision =  {
            "method": "UpdateStrategyDefault"
        }
        
        #print(self.task_update_decision,'---')
        self.task_update_decision_param = self.seen_task_learning._parse_param(
            self.task_update_decision.get("param", {})
        )

        self.config = dict(
            ll_kb_server=Context.get_parameters("KB_SERVER"),
            output_url=Context.get_parameters(
                "OUTPUT_URL",
                "/tmp"),
            cloud_output_url=Context.get_parameters(
                "OUTPUT_URL",
                "/tmp"),
            edge_output_url=Context.get_parameters(
                "EDGE_OUTPUT_URL",
                KBResourceConstant.EDGE_KB_DIR.value),
            task_index=KBResourceConstant.KB_INDEX_NAME.value)

        # self.cloud_knowledge_management = CloudKnowledgeManagement(
        #     config, estimator=e)
        #
        # self.edge_knowledge_management = EdgeKnowledgeManagement(
        #     config, estimator=e)
        #
        # self.unseen_task_processing = UnseenTaskProcessing(
        #     estimator,
        #     config,
        #     self.cloud_knowledge_management,
        #     self.edge_knowledge_management,
        #     unseen_task_allocation)

        self.unseen_task_processing = UnseenTaskProcessing(
            estimator,
            self.config,
            unseen_task_allocation)

        # task_index = FileOps.join_path(self.config['output_url'],
        #                                KBResourceConstant.KB_INDEX_NAME.value)
        #
        # self.config['task_index'] = task_index
        super(LifelongLearning, self).__init__(
            estimator=self.seen_task_learning, config=self.config
        )
        self.job_kind = K8sResourceKind.LIFELONG_JOB.value
        self.kb_server = KBClient(kbserver=self.config['ll_kb_server'])
        if self.local_test:
            self.kb_server = None

    def get_cloud_knowledge_management(self):
        return CloudKnowledgeManagement(self.config, estimator=self.seen_task_learning)

    def get_edge_knowledge_management(self):
        return EdgeKnowledgeManagement(self.config, estimator=self.estimator)

    def train(self, train_data, valid_data=None, post_process=None, **kwargs):
        has_completed_initial_training = str(Context.get_parameters("HAS_COMPLETED_INITIAL_TRAINING", "false")).lower()
        if has_completed_initial_training == "false":
            return self._initial_train(train_data, valid_data=valid_data, post_process=post_process, **kwargs)

        return self._update(train_data, valid_data=valid_data, post_process=post_process, **kwargs)

    def _initial_train(self, train_data, valid_data=None, post_process=None, **kwargs):
        """
        fit for update the knowledge based on training data.

        Parameters
        ----------
        train_data : BaseDataSource
            Train data, see `sedna.datasources.BaseDataSource` for more detail.
        valid_data : BaseDataSource
            Valid data, BaseDataSource or None.
        post_process : function
            function or a registered method, callback after `estimator` train.
        action : str
            `update` or `initial` the knowledge base
        kwargs : Dict
            parameters for `estimator` training, Like:
            `early_stopping_rounds` in Xgboost.XGBClassifier

        Returns
        -------
        train_history : object
        """

        callback_func = None
        if post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        import pdb
        # pdb.set_trace()

        # 跳到/data/user8302433/anaconda3/envs/new_ianvs/lib/python3.8/site-packages/sedna/algorithms/seen_task_learning/seen_task_learning.py(404)
        res, seen_task_index = self.estimator.train(
            train_data=train_data,
            valid_data=valid_data,
            **kwargs
        )  # todo: Distinguishing incremental update and fully overwrite
        # pdb.set_trace()
        
        # res：{'semantic_segamentation_model': {}}
        # seen_task_index：{'extractor': {'all': 0}, 'task_groups': [<sedna.algorithms.seen_task_learning.artifact.TaskGroup object at 0x7fe700350550>]}

        unseen_res, unseen_task_index = self.unseen_task_processing.initialize()
        # unseen_res:{},   unseen_task_index: {'extractor': None, 'task_groups': []}

        task_index = dict(
            seen_task=seen_task_index,
            unseen_task=unseen_task_index)
        # task_index:  {'seen_task': {'extractor': {'all': 0}, 'task_groups': [<sedna.algorithms.seen_task_learning.artifact.TaskGroup object at 0x7fe700350550>]}, 'unseen_task': {'extractor': None, 'task_groups': []}}

        cloud_knowledge_management = self.get_cloud_knowledge_management()
        task_index_url = FileOps.dump(
            task_index, cloud_knowledge_management.local_task_index_url)

        task_index = cloud_knowledge_management.update_kb(
            task_index_url, self.kb_server, local_test=self.local_test)

        if self.local_test:
            print("Lifelong learning Train task Finished, KB index save in ", task_index)
            # task_index: './workspace/benchmarkingjob/rfnet_lifelong_learning/d7c3998e-135c-11ee-be1c-45d24106221e/output/train/1/index.pkl'
            return task_index

        res.update(unseen_res)

        task_info_res = self.estimator.model_info(
            task_index,
            relpath=self.config.data_path_prefix)
        self.report_task_info(
            None, K8sResourceKindStatus.COMPLETED.value, task_info_res)
        self.log.info(f"Lifelong learning Train task Finished, "
                      f"KB index save in {task_index}")
        
        return callback_func(self.estimator, res) if callback_func else res

    def _update(self, train_data, valid_data=None, post_process=None, **kwargs):
        """
        fit for update the knowledge based on incremental data.

        Parameters
        ----------
        train_data : BaseDataSource
            Train data, see `sedna.datasources.BaseDataSource` for more detail.
        valid_data : BaseDataSource
            Valid data, BaseDataSource or None.
        post_process : function
            function or a registered method, callback after `estimator` train.
        kwargs : Dict
            parameters for `estimator` training, Like:
            `early_stopping_rounds` in Xgboost.XGBClassifier

        Returns
        -------
        train_history : object
        """
        callback_func = None
        if post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        cloud_knowledge_management = self.get_cloud_knowledge_management()
        task_index_url = self.get_parameters(
            "CLOUD_KB_INDEX", cloud_knowledge_management.task_index)
        index_url = cloud_knowledge_management.local_task_index_url
        FileOps.download(task_index_url, index_url)
        
        import pdb
        # pdb.set_trace()

        unseen_sample_re_recognition = ClassFactory.get_cls(ClassType.UTD, self.unseen_sample_re_recognition["method"])(index_url, **self.unseen_sample_re_recognition_param)

        seen_samples, unseen_samples = unseen_sample_re_recognition(train_data) # 对已经划分过的样本（unseen）再进行一次划分
            # train_data：7张，seen_samples：3张，unseen_samples：4张
        # TODO: retrain temporarily
        # historical_data = self._fetch_historical_data(index_url)
        # seen_samples.x = np.concatenate(
        #     (historical_data.x, seen_samples.x, unseen_samples.x), axis=0)
        # seen_samples.y = np.concatenate(
        #     (historical_data.y, seen_samples.y, unseen_samples.y), axis=0)

        seen_samples.x = np.concatenate( # 划分出来以后又合并？
            (seen_samples.x, unseen_samples.x), axis=0)
        seen_samples.y = np.concatenate(
            (seen_samples.y, unseen_samples.y), axis=0) # 最后全部又都放到了seen_samples

        task_update_decision = ClassFactory.get_cls(
            ClassType.KM, self.task_update_decision["method"])(
            index_url, **self.task_update_decision_param) 

        #####
        tasks, task_update_strategies = task_update_decision( # 确定每一种任务的更新策略
            seen_samples, task_type="seen_task") # 为每一个initial_train时创造的任务分配当前round样本——类似于task_allocation的操作
        seen_task_index = cloud_knowledge_management.estimator.update( # 执行每一种任务的更新策略 #跳到seentasklearning的update里
            tasks, task_update_strategies, task_index=index_url) # 基于上一步分配的样本，对每个任务的对应模型进行训练

        # seen_task_index怎么变sim、real了？unseen_task_processing.update做了什么？为啥这里还要把unseen分为seen和unseen，它们的作用是什么？

        #####
        tasks, task_update_strategies = task_update_decision(
            unseen_samples, task_type="unseen_task")
        unseen_task_index = self.unseen_task_processing.update(
            tasks, task_update_strategies, task_index=index_url)

        task_index = {
            "seen_task": seen_task_index,
            "unseen_task": unseen_task_index,
        }
        # {'seen_task': {'extractor': {'real': 0, 'sim': 1}, 'task_groups': [<sedna.algorithms.seen_task_learning.artifact.TaskGroup object at 0x7efef0af0a90>]}, 'unseen_task': {'extractor': None, 'task_groups': []}}

        task_index = cloud_knowledge_management.update_kb(
            task_index, self.kb_server, local_test=self.local_test)

        if self.local_test:
            return task_index # './workspace/benchmarkingjob/rfnet_lifelong_learning/c330a7bc-1ade-11ee-ba17-a906087290a8/output/train/2/index.pkl'

        task_info_res = self.estimator.model_info(
            task_index,
            relpath=self.config.data_path_prefix)

        self.report_task_info(
            None, K8sResourceKindStatus.COMPLETED.value, task_info_res)
        self.log.info(f"Lifelong learning Update task Finished, "
                      f"KB index save in {task_index}")
        return callback_func(self.estimator,
                             task_index) if callback_func else task_index

    def evaluate(self, data, post_process=None, **kwargs):
        """
        evaluated the performance of each task from training, filter tasks
        based on the defined rules.

        Parameters
        ----------
        data : BaseDataSource
            valid data, see `sedna.datasources.BaseDataSource` for more detail.
        kwargs: Dict
            parameters for `estimator` evaluate, Like:
            `ntree_limit` in Xgboost.XGBClassifier
        """

        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        cloud_knowledge_management = self.get_cloud_knowledge_management()
        task_index_url = Context.get_parameters(
            "MODEL_URLS", cloud_knowledge_management.task_index)
        index_url = cloud_knowledge_management.local_task_index_url

        import pdb
        # pdb.set_trace()
        self.log.info(
            f"Download kb index from {task_index_url} to {index_url}")
        FileOps.download(task_index_url, index_url)

        
        res, index_file = self._task_evaluation(
            cloud_knowledge_management, data, task_index=index_url, **kwargs)
        # res, index_file = self._task_evaluation(
        #     cloud_knowledge_management, data, task_index=task_index_url, **kwargs)
        self.log.info("Task evaluation finishes.")
        # cloud_knowledge_management.task_index: './workspace/benchmarkingjob/rfnet_lifelong_learning/c42936c2-1361-11ee-be1c-45d24106221e/output/eval/1/index.pkl'
        FileOps.upload(index_file, cloud_knowledge_management.task_index) # 实际上就是返回dst的地址，并删除原来的文件（src）。即用cloud_knowledge_management.task_index覆盖index_file内容
        self.log.info(
            f"upload kb index from {index_file} to {cloud_knowledge_management.task_index}")

        if self.local_test:
            return cloud_knowledge_management.task_index

        task_info_res = self.estimator.model_info(
            cloud_knowledge_management.task_index, result=res,
            relpath=self.config.data_path_prefix)
        self.report_task_info(
            None,
            K8sResourceKindStatus.COMPLETED.value,
            task_info_res,
            kind="eval")
        return callback_func(res) if callback_func else res

    def inference(self, data=None, post_process=None, **kwargs):
        """
        predict the result for input data based on training knowledge.

        Parameters
        ----------
        data : BaseDataSource
            inference sample, see `sedna.datasources.BaseDataSource` for
            more detail.
        post_process: function
            function or a registered method,  effected after `estimator`
            prediction, like: label transform.
        kwargs: Dict
            parameters for `estimator` predict, Like:
            `ntree_limit` in Xgboost.XGBClassifier

        Returns
        -------
        """
        self.cloud_knowledge_management = self.get_cloud_knowledge_management()
        self.edge_knowledge_management = self.get_edge_knowledge_management()
        res, tasks, is_unseen_task = None, [], False
        task_index_url = Context.get_parameters(
            "MODEL_URLS", self.cloud_knowledge_management.task_index)
        index_url = self.edge_knowledge_management.task_index # 这个地址在local测试中有误，不能被直接使用，不同于evaluate()

        import pdb
        # pdb.set_trace()
        # task_index_url， index_url： ('./workspace/benchmarkingjob/rfnet_lifelong_learning/c59e1986-1367-11ee-be1c-45d24106221e/output/eval/1/index.pkl', '/var/lib/sedna/kb/index.pkl')
        if not FileOps.exists(index_url):          
            FileOps.download(task_index_url, index_url)
            self.log.info(
                f"Download kb index from {task_index_url} to {index_url}")

            self.edge_knowledge_management.update_kb(index_url)
            self.log.info(f"Tasks are deployed at the edge.")

        unseen_sample_recognition = ClassFactory.get_cls(     # /data/user8302433/anaconda3/envs/new_ianvs/lib/python3.8/site-packages/sedna/algorithms/unseen_task_detection/unseen_sample_recognition/unseen_sample_recognition.py(27)__call__()
            ClassType.UTD,
            self.unseen_sample_recognition["method"])(
            self.edge_knowledge_management.task_index,
            **self.unseen_sample_recognition_param)

        seen_samples, unseen_samples = unseen_sample_recognition(
            data, **kwargs) # 先划分seen和unseen样本，再在两组样本下进一步各自划分多个任务 # 目前的划分方法是随机划分（一半一半概率）

        import pdb
        # pdb.set_trace()

        # 当前版本每次data都只包含一个样本（data.num_examples()==1），所以其要么是unseen的要么是seen的。但是我不太知道对unseen data的预测为什么选取的是一个未训练的初始模型，这会影响results。
        # 所以，我将对unseen data的predict也替换成了seen data的predict。
        # 注意：以上过程并不会影响unseen_sample_recognition，即不会影响当前样本是否是unseen的判断。
        # 对未知样本进行predict-------------------------------------------------------------------
        if unseen_samples.x is not None and len(unseen_samples.x) > 0:
            self.edge_knowledge_management.log.info(
                f"Unseen task is detected.")

            # 原来的unseen data predict
            # unseen_res, unseen_tasks = self.unseen_task_processing.predict(
            #     self.edge_knowledge_management,
            #     unseen_samples,
            #     task_index=self.edge_knowledge_management.task_index)
            # 替换成seen data predict一样的模式，这只会影响results，其它的都不会影响
            unseen_res, unseen_tasks = self.edge_knowledge_management.estimator.predict(
                data=unseen_samples, post_process=post_process,
                task_index=task_index_url,
                task_type="unseen_task",
                **kwargs
            )

            self.edge_knowledge_management.save_unseen_samples(unseen_samples, post_process=post_process)

            res = unseen_res
            tasks.extend(unseen_tasks)
            if data.num_examples() == 1:
                is_unseen_task = True
            else:
                image_names = list(map(lambda x: x[0], unseen_samples.x))
                is_unseen_task = dict(zip(image_names, [True] * unseen_samples.num_examples())) # 意思是部分样本是unseen，所以将unseen样本标记一下，但是当前data.num_examples()一直是1

        # 对已知样本进行predict---------------------------------------------------------------------
        if seen_samples.x is not None and len(seen_samples.x) > 0:        
            # seen_res, seen_tasks = self.edge_knowledge_management.estimator.predict(
            #     data=seen_samples, post_process=post_process,
            #     task_index=index_url,
            #     task_type="seen_task",
            #     **kwargs
            # )
            seen_res, seen_tasks = self.edge_knowledge_management.estimator.predict(
                data=seen_samples, post_process=post_process,
                task_index=task_index_url, # local测试时不能用index_url
                task_type="seen_task",
                **kwargs
            )

            res = np.concatenate((res, seen_res)) if res else seen_res
            tasks.extend(seen_tasks) # seen_tasks: [<sedna.algorithms.seen_task_learning.artifact.Task object at 0x7fc95caa6670>];    seen_tasks[0].model.meta_attr: ['all']
            if data.num_examples() > 1:
                image_names = list(map(lambda x: x[0], seen_samples.x))
                is_unseen_dict = dict(zip(image_names, [False] * seen_samples.num_examples()))
                if isinstance(is_unseen_task, bool):
                    is_unseen_task = is_unseen_dict
                else:
                    is_unseen_task.update(is_unseen_dict)

        return res, is_unseen_task, tasks

    def _task_evaluation(self, cloud_knowledge_management, data, **kwargs):
        import pdb
        # pdb.set_trace()
        res, tasks_detail = cloud_knowledge_management.estimator.evaluate(
            data=data, **kwargs)
        drop_task = cloud_knowledge_management.evaluate_tasks(
            tasks_detail, **kwargs)

        index_file = None
        if not self.local_test:
            index_file = self.kb_server.update_task_status(drop_task, new_status=0)

        if not index_file:
            index_file = str(cloud_knowledge_management.local_task_index_url)
            index_file = self._drop_task_from_index(drop_task, index_file)
        else:
            self.log.info(f"Deploy {index_file} to the edge.")

        return res, index_file

    def _drop_task_from_index(self, drop_task, index_file):
        """

        Parameters
        ----------
        drop_tasks: str, e,g,: "task_entry1, task_entry2"
        index_file: str, e,g,: index.pkl

        Returns: str, e,g,: index.pkl
        -------

        """
        task_dict = FileOps.load(index_file)

        def _drop_task(temp):
            for task_group in temp[KBResourceConstant.TASK_GROUPS.value]:
                for task in task_group.tasks:
                    if task.entry in drop_task.split(","):
                        task_group.tasks.remove(task)

        if drop_task:
            _drop_task(task_dict[KBResourceConstant.SEEN_TASK.value])
            _drop_task(task_dict[KBResourceConstant.UNSEEN_TASK.value])
            task_dict['create_time'] = str(time.time())
            FileOps.dump(task_dict, index_file)

        return index_file

    def _fetch_historical_data(self, task_index):
        if isinstance(task_index, str):
            task_index = FileOps.load(task_index)

        samples = BaseDataSource(data_type="train")

        for task_group in task_index["seen_task"]["task_groups"]:
            if isinstance(task_group.samples, BaseDataSource):
                _samples = task_group.samples
            else:
                _samples = FileOps.load(task_group.samples.data_url)

            samples.x = _samples.x if samples.x is None else np.concatenate(
                (samples.x, _samples.x), axis=0)
            samples.y = _samples.y if samples.y is None else np.concatenate(
                (samples.y, _samples.y), axis=0)

        return samples
