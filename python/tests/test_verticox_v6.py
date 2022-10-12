import vantage6.client as v6client
from clize import run

from verticox.client import VerticoxClient

IMAGE = 'harbor.carrier-mu.src.surf-hosted.nl/carrier/verticox'


def run_verticox_v6(host, port, user, password, private_key, tag='latest'):
    image = f'{IMAGE}:{tag}'

    client = v6client.Client(host, port)

    client.authenticate(user, password)
    client.setup_encryption(private_key)
    nodes = client.node.list(is_online=True)

    orgs = [n['id'] for n in nodes['data']]
    central_node = orgs[0]
    datanodes = orgs[1:]

    verticox_client = VerticoxClient(client, image=image)

    task = verticox_client.get_column_names()

    print(task.get_result())

    feature_columns = ['afb', 'age', 'gender']

    task = verticox_client.compute(feature_columns, 'event_time', 'event_happened',
                                   datanodes=datanodes, central_node=central_node)

    results = task.get_result(timeout=10 * 60)
    for result in results:
        print(f'Organization: {result.organization}')
        print(f'Log: {result.log}')
        print(f'Content: {result.content}')


if __name__ == '__main__':
    run(run_verticox_v6)
