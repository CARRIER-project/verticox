import json
import sys
from typing import List, Optional, Union

import requests
from requests.exceptions import ConnectionError
from vantage6.common import debug
from vantage6.tools.util import info

_DEFAULT_PRECISION = 5
_SET_PRECISION = 'setPrecisionCentral'
_SUM_RELEVANT_VALUES = 'sumRelevantValues'
_INIT_CENTRAL_SERVER = 'initCentralServer'
_PROTOCOL = 'http://'
_OK = 200
_SET_ENDPOINTS = 'setEndpoints'
_SET_ID = 'setID'
_LONG_TIMEOUT = 60
_COMMODITY_ID = 'commodity'
_MAX_T_VALUE = 'inf'


# TODO: Split up in datanode client and aggregator client
class NPartyScalarProductClient:

    def __init__(self, commodity_address: str, external_commodity_address=None,
                 other_addresses: Optional[List[str]] = None,
                 precision: Optional[int] = _DEFAULT_PRECISION):
        """

        Args:
            commodity_address:
            other_addresses:
            precision:
        """
        self._internal_address = commodity_address
        self._external_commodity_address = external_commodity_address
        self.other_addresses = other_addresses
        self.precision = precision

        info(f'Initializing N party client with:\n'
             f'Internal commodity address: {self._internal_address}\n'
             f'External commodity address: {self._external_commodity_address}\n'
             f'Scalar product Datanode addresses: {self.other_addresses}\n'
             f'Precision: {self.precision}')

    def initialize_servers(self):
        self._init_central_server(self._internal_address, self.other_addresses)
        info('Initialized central server')

        self._init_datanodes(self._external_commodity_address, self.other_addresses)
        info('Initialized datanodes')

        # # Setting endpoints for central server
        # self._put_endpoints(self._internal_address, self.other_addresses)
        # info('Specified endpoints for central server')

        debug('Setting precision')
        self._set_precision(self.precision)
        debug('Done setting precision')

    # TODO: Make sure terminology is consistent over all code
    def sum_relevant_values(self, numeric_features: List[str], boolean_feature: str,
                            boolean_value: bool) -> List[int]:
        all_sums = []
        for feature in numeric_features:
            parameters = {
                'requirements': [{
                    'value': {
                        'type': 'bool',
                        'value': boolean_value,
                        'attributeName': boolean_feature,
                    },
                    'range': False
                }],
                'predictor': feature
            }
            info(f'Sum relevant values with parameters: {parameters}')
            result = self._post(_SUM_RELEVANT_VALUES, json=parameters)
            all_sums.append(result)

        return all_sums

    def sum_records_at_risk(self, values: List[Union[float]], t: float, t_column: str) -> \
            float:
        """
        n-scalar version of the following component of the central computation:

        1. `value` will be sent to the java commodity server
        2. n-scalar protocol will be used to sum all z values belonging to a record that is at
            risk at time t. This is done by only including records with a event time >= t.

        Args:
            t_column:
            t:
            z:

        Returns:

        """
        print(f't: {t}')

        # TODO: Send z values
        post_z_params= {'z': values, 'attribute': t_column}
        print(f'Post z params: {[post_z_params]}')
        self._post('postZ', json=post_z_params)

        # TODO: Compute sum of z values
        parameters = {
            "requirements": [{
                "range": True,
                "upperLimit": {'type': 'real', 'value': 'inf', 'attributeName': t_column},
                "lowerLimit": {'type': 'real', 'value': t, 'attributeName': t_column}}
            ],
        }

        print(f'Sum z parameters: {parameters}')
        result =  self._post('sumZValues', json=parameters)

        return float(result)

    def get_relevant_values(self):
        pass

    def kill_nodes(self):
        nodes_to_kill = self.other_addresses + [self._internal_address]

        for n in nodes_to_kill:
            try:
                self._kill_endpoint(n)
            except ConnectionError:
                # A connection error means that the node has successfully shut down (most likely)
                pass

    def _init_datanodes(self, commodity_address, other_addresses):
        for idx, address in enumerate(other_addresses):
            # Set id first
            datanode_id = f'datanode_{idx}'
            set_id_endpoint = f'http://{address}/{_SET_ID}'
            requests.post(set_id_endpoint, params={'id': datanode_id})
            others = other_addresses.copy()
            others.remove(address)
            others.append(commodity_address)
            self._put_endpoints(address, others)

    def _set_precision(self, precision):
        params = {'precision': precision}
        self._put(_SET_PRECISION, params=params)

    def _init_central_server(self, internal_address, other_nodes):
        other_nodes = [f'{_PROTOCOL}{n}' for n in other_nodes]
        json = {'secretServer': f'{_PROTOCOL}{internal_address}', 'servers': other_nodes}
        debug(f'Initializing central server with: {json}')
        self._post(_INIT_CENTRAL_SERVER,
                   json=json, timeout=_LONG_TIMEOUT)

        self._post(_SET_ID, params={'id': _COMMODITY_ID})

    def _request(self, method, endpoint, **kwargs):
        url = self._get_url(endpoint)
        result = requests.request(method, url, **kwargs)

        self.check_response_code(result)

        return result.content

    def _post(self, endpoint, **kwargs):
        return self._request('post', endpoint, **kwargs)

    def _get(self, endpoint, **kwargs):
        return self._request('get', endpoint, **kwargs)

    def _put(self, endpoint, **kwargs):
        return self._request('put', endpoint, **kwargs)

    @staticmethod
    def check_response_code(result):
        if not result.ok:
            raise Exception(f'Received response code {result.status_code}')

    def _get_url(self, endpoint, address=None):
        if address is None:
            address = self._internal_address
        return f'{_PROTOCOL}{address}/{endpoint}'

    def _put_endpoints(self, targetUrl, others):
        others = [f'{_PROTOCOL}{o}' for o in others]
        payload = {"servers": others}
        debug(f'Setting endpoints for {targetUrl} with: {payload}')
        url = f'{_PROTOCOL}{targetUrl}/{_SET_ENDPOINTS}'
        requests.post(url, json=payload, timeout=10)

    def _kill_endpoint(self, target):
        r = requests.put(target + "/kill")


def main():
    datanodes = ['datanode1', 'datanode2']
    commodity_node = 'commodity'
    client = NPartyScalarProductClient(commodity_node, datanodes)

    client.sum_relevant_values('x3', 'x6', 1)
    # TODO: Implement postz and sumzvalues

    client.kill_nodes()

    info('All steps have run succesfully')


if __name__ == '__main__':
    main()

#
# localhost: 8080 / setPrecisionCentral?precision = 5
#
# {
#     'value': null,
#     'range': true,
#     'upperLimit': {
#         'type': 'numeric',
#         'value': '10',
#         'attributeName': 'x1',
#         'id': null,
#         'uknown': false
#     },
#     'lowerLimit': {
#         'type': 'numeric',
#         'value': '-inf',
#         'attributeName': 'x1',
#         'id': null,
#         'uknown': false
#     },
#     'name': 'x1'
# }
#
# // If
# value is used
# it is a
# direct
# comparison, so
# requirement is fullfilled if value == attribute
# // If
# range is used
# then
# lower <= attribue < upperLimit
