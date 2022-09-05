import vantage6.client

IMAGE = 'harbor.carrier-mu.src.surf-hosted.nl/carrier/verticox_predictors'
NAME = 'verticox_predictors from client'


class VerticoxPredictorsClient:

    def __init__(self, client: vantage6.client.Client):
        """

        :param client: Vantage6 client
        """
        self.client = client

    def verticox(self, collaboration, commodity_node, nodes, requirements,predictors):
        return self.client.task.create(collaboration=collaboration,
                                       organizations=[commodity_node],
                                       name=NAME, image=IMAGE, description=NAME,
                                       input={'method': 'verticoxPredictors', 'master': True,
                                              'args': [nodes, requirements, predictors]})
