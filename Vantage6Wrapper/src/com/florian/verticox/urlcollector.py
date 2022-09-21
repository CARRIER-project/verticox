import requests
from typing import Any, Dict, Tuple

from requests.adapters import HTTPAdapter
from urllib3 import Retry
from vantage6.tools.util import info
from time import sleep
import traceback

RETRY = 10

#
# def shareEndpoints(client, data, exclude_orgs=None, **kwargs):
#     try:
#         return _try_set_endpoints(client, exclude_orgs, kwargs.get("task_id"))
#     except Exception as e:
#         info('Exception!')
#         info(traceback.format_exc())
#         raise e
#
#
# def killEndpoints(client, data, exclude_orgs=None, **kwargs):
#     try:
#         _try_kill_endpoints(client, exclude_orgs, kwargs.get("task_id"))
#     except Exception as e:
#         info('Excetpion!')
#         info(traceback.format_exc())
#         raise e
#
#
# def getEndpoints(client, data, exclude_orgs=None, **kwargs):
#     try:
#         _try_get_endpoints(client, exclude_orgs, kwargs.get("task_id"))
#     except Exception as e:
#         info('Excetpion!')
#         info(traceback.format_exc())
#         raise e
#
#
# def _try_get_endpoints(client, exclude_orgs, task_id):
#     task = client.get_task(task_id)
#
#     # Ip address and port of algorithm can be found in results model
#     return _await_port_numbers(client, task.get('id'))
#
#
# def _try_kill_endpoints(client, exclude_orgs, task_id):
#     task = client.get_task(task_id)
#
#     # Ip address and port of algorithm can be found in results model
#     result_objects = _await_port_numbers(client, task.get('id'))
#
#     for i in range(len(result_objects)):
#         target = result_objects[i]
#         _kill_endpoint(target)

#
# def _try_set_endpoints(client, exclude_orgs, task_id):
#     task = client.get_task(task_id)
#
#     # Ip address and port of algorithm can be found in results model
#     result_objects = _await_port_numbers(client, task.get('id'))
#
#     for i in range(len(result_objects)):
#         target = result_objects[i]
#         others = result_objects[~i]
#
#         info(f'Sending message')
#         _set_endpoints(target, others)


def _get_address_from_result(result: Dict[str, Any]) -> Tuple[str, int]:
    address = result['ip']
    port = result['port']

    return address, port


def _set_endpoints(target, endpoints):
    target_ip, target_port = _get_address_from_result(target)
    others = []
    targetUrl = "http://" + target_ip + ":" + target_port
    for e in endpoints:
        other_ip, other_port = _get_address_from_result(e)
        others.append("http://" + other_ip + ":" + other_port)

    return put_endpoints(others, targetUrl)


def put_endpoints(targetUrl, others):
    payload = {"servers": others}
    r = requests.post(targetUrl + "/setEndpoints", json=payload, timeout=10)



def _kill_endpoint(target):
    target_ip, target_port = _get_address_from_result(target)
    targetUrl = "http://" + target_ip + ":" + target_port
    r = requests.put(targetUrl + "/kill")


def _get_connection_session():
    """
    Create connection adapter with sensible retry strategy
    :return:
    """
    retry_strategy = Retry(
        total=RETRY
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    return http
