import json
import re

import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

BRACKETS = re.compile(r'[\[\]]')
BETA_PATTERN = re.compile(
    r'DEBUG:verticox.datanode :: institution no. (?P<institution_number>\d+):Beta: (?P<array>.*)')

LOG_FILE = 'log.txt'
LZ_LINE_START = 'DEBUG:verticox.aggregator:Lz: '
LZ_VALUE_POSITION = len(LZ_LINE_START)
Z_DIFF_LINE_START = 'DEBUG:verticox.aggregator:z_diff: '
Z_DIFF_VALUE_POSITION = len(Z_DIFF_LINE_START)
SIGMA_DIFF_LINE_START = 'DEBUG:verticox.aggregator:sigma_diff: '
SIGMA_DIFF_VALUE_POSITION = len(SIGMA_DIFF_LINE_START)

TARGET_BETA_START = 'INFO:__main__:Target result: '
TARGET_BETA_POSITION = len(TARGET_BETA_START)
MAX_INSTITUTIONS = 5

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

_log_file = None

lz = []
z_diff = []
sigma_diff = []
beta = {}
mse = []
target_beta = {}

fig = make_subplots(rows=4, cols=1, row_heights=[20] * 4,
                    subplot_titles=['Lz', 'z diff', 'sigma diff', 'Mean squared error'])

fig.add_trace({'y': lz, 'name': 'Lz', 'mode': 'lines+markers', 'type': 'scatter'}, row=1, col=1)
fig.add_trace({'y': z_diff, 'name': 'z diff', 'mode': 'lines+markers', 'type': 'scatter'},
              row=2, col=1)

fig.add_trace({'y': sigma_diff,
               'name': 'sigma diff',
               'mode': 'lines+markers',
               'type': 'scatter'}, row=3, col=1)

fig.add_trace({'y': [], 'name': 'Mean Squared Error', 'mode': 'lines+markers'}, row=4, col=1)

fig.update_layout(height=1000, width=1500)

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
    filter_lines()

    fig.update_traces({'y': lz}, row=1, col=1)
    fig.update_traces({'y': z_diff}, row=2, col=1)
    fig.update_traces({'y': sigma_diff}, row=3, col=1)
    fig.update_traces({'y': mse}, row=4, col=1)

    return fig


def get_log_file():
    global _log_file
    if _log_file is None:
        _log_file = open(LOG_FILE, 'r')

    return _log_file


def filter_lines():
    lines = get_log_file().readlines()

    for line in lines:
        if line.startswith(LZ_LINE_START):
            lz.append(float(line[LZ_VALUE_POSITION:]))
        elif line.startswith(Z_DIFF_LINE_START):
            z_diff.append(float(line[Z_DIFF_VALUE_POSITION:]))
        elif line.startswith(SIGMA_DIFF_LINE_START):
            sigma_diff.append(float(line[SIGMA_DIFF_VALUE_POSITION:]))
        elif line.startswith(TARGET_BETA_START):
            global target_beta
            print(line[TARGET_BETA_POSITION:])
            target_beta = np.array(json.loads(line[TARGET_BETA_POSITION:]))
        else:
            m = BETA_PATTERN.match(line)
            if m:
                variables = m.groupdict()
                new_beta = np.array(json.loads(variables['array']))
                institution_number = int(variables['institution_number'])

                if institution_number not in beta.keys():
                    beta[institution_number] = []
                beta[institution_number].append(new_beta)

                new_mse = compute_mse()

                print(f'New mse: {new_mse}')

                mse.append(new_mse)


def compute_mse():
    sum_of_squares = 0
    num_items = 0
    for institution, values in beta.items():
        last_value = values[-1]
        sum_of_squares += np.square(target_beta[institution] - last_value).sum()
        num_items += len(values)

    return sum_of_squares / num_items


#
# def compute_mse(beta):
#     if target_beta is not None:
#         return np.square(beta - target_beta).mean()


if __name__ == '__main__':
    app.run_server(debug=True)
