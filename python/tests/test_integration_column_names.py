import vantage6.client as v6client
from clize import run

from verticox.client import VerticoxClient
from verticox.config import DOCKER_IMAGE


def run_column_names_v6(host, port, user, password, private_key, tag="latest"):
    image = f"{DOCKER_IMAGE}:{tag}"

    client = v6client.Client(host, port)

    client.authenticate(user, password)
    client.setup_encryption(private_key)
    nodes = client.node.list(is_online=True)

    orgs = [n["id"] for n in nodes["data"]]
    central_node = orgs[0]
    datanodes = orgs[1:]

    verticox_client = VerticoxClient(client, image=image)

    task = verticox_client.get_column_names()

    print(task.get_results())


if __name__ == "__main__":
    run(run_column_names_v6)
