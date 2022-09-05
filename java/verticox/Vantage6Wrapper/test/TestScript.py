import vantage6.client as v6client

from com.florian.verticox.wrapper.client import VerticoxPredictorsClient

PRIVATE_KEY_DIR= "C:\Florian\GIT\KEY\CARRIERSurfKey.pem"
USERNAME="florian"
PASSWORD="1Lv*1Bt*uG4!1ogX"


HOST = 'https://v6server.carrier-mu.src.surf-hosted.nl'
PORT = 443
PRIVATE_KEY = PRIVATE_KEY_DIR

client = v6client.Client(host=HOST, port=PORT, verbose=False)
client.authenticate('florian', '1Lv*1Bt*uG4!1ogX')

client.setup_encryption(PRIVATE_KEY)

COLUMN_NAMES_IMAGE = 'harbor2.vantage6.ai/testing/v6-test-py'

verticox = VerticoxPredictorsClient(client)

node1 = 3
node2 = 5
commodity_node = 2
exclude = [4,6]

collaboration_id=1

predictors =["x2","x3"]

requirements =[{
    "value": {
      "type": "numeric",
      "value": "1",
      "attributeName": "x1",
      "uknown": False
    },
    "range": False,
    "name": "x1"
  }]


bins = False
task = verticox.verticox(collaboration_id, commodity_node, [node1, node2], requirements, predictors   )

done = False


print(client.node.list(is_online=True))

nodes = client.node.list(is_online=True)
# TODO: Add pagination support
nodes = nodes['data']
xsad = [n['organization']['id'] for n in nodes]
print(xsad)

while(not done):
    for r in task['results']:
        updated = client.result.get(r['id'])
        organization = updated["organization"]["id"]
        result = updated['result']
        if result != None:
            done = True