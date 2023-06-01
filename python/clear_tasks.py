import clize
from vantage6.client import Client


def run_verticox_v6(host, port, user, password, private_key):
    client = Client(host, port)
    client.authenticate(user, password)
    client.setup_encryption(private_key)

    while True:
        tasks = client.task.list()

        if len(tasks['data']) == 0:
            break

        print(f'Deleting {len(tasks["data"])} tasks')

        for t in tasks['data']:
            client.task.delete(t['id'])


if __name__ == '__main__':
    clize.run(run_verticox_v6)
