import dash
import plotly.express as px
from plotly.subplots import make_subplots
from dash import dcc, html
from dash.dependencies import Input, Output
from multiprocessing import Lock

LOG_FILE = 'log.txt'
LZ_LINE_START = 'DEBUG:verticox.aggregator:Lz: '
LZ_VALUE_POSITION = len(LZ_LINE_START)
Z_DIFF_LINE_START = 'DEBUG:verticox.aggregator:z_diff: '
Z_DIFF_VALUE_POSITION = len(Z_DIFF_LINE_START)
SIGMA_DIFF_LINE_START = 'DEBUG:verticox.aggregator:sigma_diff: '
SIGMA_DIFF_VALUE_POSITION = len(SIGMA_DIFF_LINE_START)

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
    new_lz, new_z_diff, new_sigma_diff = list(filter_lines())

    print(new_sigma_diff)

    fig = make_subplots(rows=3, cols=1, row_heights=[20] * 3, subplot_titles=[
        'Lz', 'z diff',
        'sigma diff'])

    # Create the graph with subplots
    fig.add_trace({'y': new_lz, 'name': 'Lz', 'mode': 'lines+markers', 'type': 'scatter'})
    fig.add_trace({'y': new_z_diff, 'name': 'z diff', 'mode': 'lines+markers', 'type': 'scatter'},
                  row=2, col=1)

    fig.add_trace({'y': new_sigma_diff,
                   'name': 'sigma diff',
                   'mode': 'lines+markers',
                   'type': 'scatter'}, row=3, col=1)

    fig.update_layout(height=1000, width=1000)

    return fig


def filter_lines():
    lz = []
    z_diff = []
    sigma_diff = []

    with open(LOG_FILE, 'r') as f:
        for line in f:
            if line.startswith(LZ_LINE_START):
                lz.append(float(line[LZ_VALUE_POSITION:]))
            elif line.startswith(Z_DIFF_LINE_START):
                z_diff.append(float(line[Z_DIFF_VALUE_POSITION:]))
            elif line.startswith(SIGMA_DIFF_LINE_START):

                sigma_diff.append(float(line[SIGMA_DIFF_VALUE_POSITION:]))

    return lz, z_diff, sigma_diff


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
