import dash
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output
from multiprocessing import Lock

LOG_FILE = 'log.txt'
LZ_LINE_START = 'DEBUG:verticox.aggregator:Lz: '
LZ_VALUE_POSITION = len(LZ_LINE_START)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

_log_stream = None


def get_log_stream():
    global _log_stream
    if not _log_stream:
        _log_stream = LzStream(LOG_FILE)

    return _log_stream


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('Convergence live feed'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=5 * 1000,  # in milliseconds
            n_intervals=0
        )
    ])
)


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    new_data = list(filter_lines())

    # Create the graph with subplots
    fig = px.line(y=new_data)
    # fig.add_trace({
    #     'y': new_data,
    #     'name': 'Lz',
    #     'mode': 'lines+markers',
    #     'type': 'scatter'
    # })

    return fig


def filter_lines():
    with open(LOG_FILE, 'r') as f:
        for line in f:
            if line.startswith(LZ_LINE_START):
                yield float(line[LZ_VALUE_POSITION:])


class LzStream:

    def __init__(self, file):
        self.file = open(file, 'r')
        self.lock = Lock()

    def __del__(self):
        self.file.close()

    def read(self):
        return list(self._read_new_lines())

    def _read_new_lines(self):
        with self.lock:
            for line in self.file:

                if line.startswith(LZ_LINE_START):
                    print(line)
                    yield float(line[LZ_VALUE_POSITION:])


if __name__ == '__main__':
    app.run_server(debug=True)
