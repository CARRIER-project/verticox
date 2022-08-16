import time
from typing import List
import logging
import verticox
from vantage6.client import Client

_VERTICOX_IMAGE = 'harbor.carrier-mu.src.surf-hosted.nl/carrier/verticox'
_DATA_FORMAT = 'json'
_SLEEP = 5

class VerticoxClient:

    def __init__(self, v6client: Client, collaboration=None, log_level=logging.INFO):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self.v6client = v6client

        collaborations = self.v6client.collaboration.list()
        if len(collaborations) > 1:
            raise VerticoxClientException(f'You are in multiple collaborations, please specify '
                                          f'one of:\n {collaborations}')

        self.collaboration_id = collaborations[0]['id']

    def get_active_node_organizations(self):
        nodes = self.v6client.node.list(is_online=True)

        # TODO: Add pagination support
        nodes = nodes['data']
        return [n['organization']['id'] for n in nodes]

    def get_column_names(self):
        active_nodes = self.get_active_node_organizations()
        self._logger.debug(f'There are currently {len(active_nodes)} active nodes')

        self._run_task('column_names', master=False)

    def _run_task(self, method, master, kwargs=None, organizations=List[int],
                  database='default'):
        if kwargs is None:
            kwargs = {}
        # TODO: Construct description out of parameters
        description = ''
        name = ''
        task_input = {'method': method, 'master': True, 'kwargs': kwargs}
        task = self.v6client.task.create(collaboration=self.collaboration_id,
                                         organizations=organizations,
                                         name=name,
                                         image=_VERTICOX_IMAGE,
                                         description=description,
                                         input=task_input,
                                         data_format=_DATA_FORMAT,
                                         database=database
                                         )
        return Task(self.v6client, task)

    # def analyze(self, feature_columns, outcome_time_colum, right_censor_column, datanode_ids,
    #             precision):
    #     input_params = {'method': 'verticox', 'master': True, 'kwargs':
    #         {
    #             'feature_columns': ['afb',
    #                                 #                                        'age', 'av3', 'bmi', 'chf', 'cvd',
    #                                 #                     'diasbp', 'gender', 'hr','los', 'miord',
    #                                 #                     'mitype', 'sho',
    #                                 'sysbp'],
    #             'event_times_column': 'event_time',
    #             'event_happened_column': 'event_happened',
    #             'datanode_ids': orgs[1:],
    #             'precision': 0.1
    #         }
    #                     }


#
#
# task = client.task.create(collaboration=1, organizations=[orgs[0]], name='verticox',
#                           image=IMAGE, description='verticox test',
#                           input=input_params)

class Task:

    def __init__(self, client: Client, task_data):
        self._raw_data = task_data
        self.client=client

        self.result_ids = [r['id'] for r in task_data['results']]

    def await_result(self):
        results = {}

        result_ids = set(self.result_ids)

        while True:
            results_missing = result_ids - results.keys()
            for missing in results_missing:
                result = self.client.result.get(missing)
                if result['finished_at'] is not None:
                    results[missing] = result

            if len(results) >= len(self.result_ids):
                break
            time.sleep(_SLEEP)


class VerticoxClientException(Exception):
    pass

