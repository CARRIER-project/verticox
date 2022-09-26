from typing import List

import requests
import logging

_DEFAULT_PRECISION = 5
_SET_PRECISION = 'setPrecisionCentral'
_SUM_RELEVANT_VALUES = 'sumRelevantValues'
_INIT_CENTRAL_SERVER = 'initCentralServer'
_PROTOCOL = 'http://'
_OK = 200
_SET_ENDPOINTS = 'setEndpoints'

logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger(__name__)


class ScalarProductClient:

    def __init__(self, commodity_address: str, other_addresses: List[str],
                 precision=_DEFAULT_PRECISION):
        self._address = f'{_PROTOCOL}{commodity_address}'

        other_addresses = [f'{_PROTOCOL}{a}' for a in other_addresses]
        self.other_addresses = other_addresses

        self._init_central_server(self._address, other_addresses)
        self._init_datanodes(self._address, other_addresses)
        self._set_precision(precision)

    def _init_datanodes(self, internal_address, other_addresses):
        for address in other_addresses:
            others = other_addresses.copy()
            others.remove(address)
            others.append(internal_address)
            self.put_endpoints(address, others)

    def _set_precision(self, precision):
        params = {'precision': precision}
        self._put(_SET_PRECISION, params=params)

    def _init_central_server(self, central_server, other_nodes):
        json={'secretServer': central_server, 'servers': other_nodes}
        _logger.debug(f'Initializing central server with: {json}')
        self._post(_INIT_CENTRAL_SERVER,
                   json=json)

    # TODO: Make sure terminology is consistent over all code
    def sum_relevant_values(self, feature_name, time_name, time_value):
        parameters = {
            'requirements': [{
                'value': {
                    'type': 'numeric',
                    'value': time_value,
                    'attributeName': time_name,
                },
                'range': False
            }],
            'predictor': feature_name
        }
        _logger.debug(f'Sum relevant values parameters: {parameters}')
        result = self._post(_SUM_RELEVANT_VALUES, json=parameters)
        return result

    def sum_z_values(self):
        pass

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
            address = self._address
        return f'{address}/{endpoint}'

    def put_endpoints(self, targetUrl, others):
        payload = {"servers": others}
        url = f'{targetUrl}/{_SET_ENDPOINTS}'
        requests.post(url, json=payload, timeout=10)

    def _kill_endpoint(self, target):
        r = requests.put(target + "/kill")


def main():
    datanodes = ['datanode1', 'datanode2']

    client = ScalarProductClient('commodity', datanodes)

    client.sum_relevant_values('x3', 'x6', 1)


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
