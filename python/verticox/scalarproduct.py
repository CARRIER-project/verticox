import requests
import logging

_DEFAULT_PRECISION = 5
_SET_PRECISION = 'setPrecisionCentral'
_SUM_RELEVANT_VALUES = 'sumRelevantValues'
_INIT_CENTRAL_SERVER = 'initCentralServer'
_PROTOCOL = 'http://'
_OK = 200

_logger = logging.getLogger(__name__)


class ScalarProductClient:

    def __init__(self, address, internal_address, other_addresses, precision=_DEFAULT_PRECISION):
        self._address = address
        self._protocol = _PROTOCOL

        self._init_central_server(address, other_addresses)
        self._set_precision(precision)

    def _set_precision(self, precision):
        params = {'precision': precision}
        self._put(_SET_PRECISION, params=params)

    def _init_central_server(self, central_server, other_nodes):
        self._post(_INIT_CENTRAL_SERVER,
                   json={'secretServer': central_server, 'servers': other_nodes})

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

        result = self._post(_SUM_RELEVANT_VALUES, json=parameters)
        return result

    def sum_z_values(self):
        pass

    def _post(self, endpoint, params=None, json=None):
        _logger.info(f'POST: {endpoint}')
        url = self._get_url(endpoint)
        result = requests.post(url, params=params, json=json)

        self.check_response_code(result)

        return result.content

    @staticmethod
    def check_response_code(result):
        if not result.ok:
            raise Exception(f'Received response code {result.status_code}')

    def _put(self, endpoint, params=None):
        url = self._get_url(endpoint)
        result = requests.put(url, params=params)

        return result.content

    def _get_url(self, endpoint):
        return f'{self._protocol}{self._address}/{endpoint}'


def main():
    datanodes = [('datanode1', 80), ('datanode2', 80)]

    client = ScalarProductClient('localhost:8080', 'commodity', datanodes)

    client.sum_relevant_values('x1', 'x2', 1)


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
