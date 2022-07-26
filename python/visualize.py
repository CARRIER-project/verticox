import json
import re

import dash
import numpy as np
import pandas as pd
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BRACKETS = re.compile(r'[\[\]]')
BETA_PATTERN = re.compile(
    r'DEBUG:verticox.datanode :: institution no. (?P<institution_number>\d+):Beta: (?P<array>.*)')

LZ_PATTERN = re.compile(
    r'DEBUG:verticox.likelihood:Lz_(?P<loop>(inner|outer)): (?P<value>.*)'
)

LOG_FILE = 'log.txt'
LZ_LINE_START = 'DEBUG:verticox.aggregator:Lz_: '
Z_DIFF_LINE_START = 'DEBUG:verticox.aggregator:z_diff: '
Z_DIFF_VALUE_POSITION = len(Z_DIFF_LINE_START)
SIGMA_DIFF_LINE_START = 'DEBUG:verticox.aggregator:sigma_diff: '
SIGMA_DIFF_VALUE_POSITION = len(SIGMA_DIFF_LINE_START)

TARGET_BETA_START = 'INFO:__main__:Target result: '
TARGET_BETA_POSITION = len(TARGET_BETA_START)
MAX_INSTITUTIONS = 5

RUNNING = 'Running...'

BETA_BAR = 'betabar'
TARGET_BAR = 'targetbar'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

_log_file = None

e = 1e-5
lz = []
lz_loop = []
z_diff = []
sigma_diff = []
beta = {}
beta_df = pd.DataFrame(columns=['name', 'value', 'target'])
mae = []
target_beta = np.array([])
coefs_are_plotted = False
done = False

fig = make_subplots(rows=5, cols=1,
                    subplot_titles=['Lz', 'z diff', 'sigma diff', 'Mean absolute error',
                                    'Current coefficients'])

fig.add_trace(go.Scatter(y=lz, name='Lz', mode='lines+markers', marker={'color': lz_loop,
                                                                        'cmin': 0,
                                                                        'cmax': 1}),
              row=1, col=1)
fig.add_trace({'y': z_diff, 'name': 'z diff', 'mode': 'lines+markers', 'type': 'scatter'},
              row=2, col=1)

fig.add_trace({'y': sigma_diff,
               'name': 'sigma diff',
               'mode': 'lines+markers',
               'type': 'scatter'}, row=3, col=1)

fig.add_trace({'y': [], 'name': 'Mean Absolute Error', 'mode': 'lines+markers'}, row=4, col=1)

fig.add_bar(x=beta_df['name'], y=beta_df['value'], name=BETA_BAR, row=5,
            col=1)
fig.add_bar(x=beta_df['name'], y=beta_df['value'], name=TARGET_BAR, row=5,
            col=1)

fig.update_layout(height=1000, width=1500)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H3('Convergence live feed'),
        html.H4('Running...', id='converged'),
        html.Progress(id='progress', max=1),
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

    fig.update_traces({'y': lz, 'marker': {'color': lz_loop}}, row=1, col=1)
    fig.update_traces({'y': z_diff}, row=2, col=1)
    fig.update_traces({'y': sigma_diff}, row=3, col=1)
    fig.update_traces({'y': mae}, row=4, col=1)

    new_df = beta_df[~(beta_df['target'])]
    target_df = beta_df[beta_df['target']]

    fig.update_traces({'x': new_df['name'], 'y': new_df['value']}, selector={'name': BETA_BAR})
    fig.update_traces({'x': target_df['name'], 'y': target_df['value']},
                      selector={'name': TARGET_BAR})

    return fig


@app.callback(Output('progress', 'value'), Input('interval-component', 'n_intervals'))
def update_progress(n_intervals):
    """
    Updating progress based on how close z_diff and sigma_diff are to the value e
    Args:
        n_intervals:

    Returns:

    """
    if (z_diff[-1] != 0) and (sigma_diff != 0):
        minimum = max(np.max(sigma_diff), np.max(z_diff))
        minimum = np.log(e / minimum)

        sigma_progress = e / sigma_diff[-1]
        z_progress = e / z_diff[-1]

        progress = np.log(min(sigma_progress, z_progress))
        range = 1 - minimum

        remapped = (progress - minimum) / range

        return str(remapped)
    else:
        return '0'


@app.callback((Output('converged', 'children'), Output('interval-component', 'disabled')),
              Input('interval-component', 'n_intervals'))
def show_status(input_value):
    if done:
        df = beta_df[["name", "value"]]
        text = 'Finished! Final betas:'
        datatable = dash_table.DataTable(df.to_dict('records'),
                                         [{'name': i, 'id': i} for i in df.columns])
        return [text, datatable], True

    return RUNNING, False


def get_log_file():
    global _log_file
    if _log_file is None:
        _log_file = open(LOG_FILE, 'r')

    return _log_file


def filter_lines():
    lines = get_log_file().readlines()

    for line in lines:
        if m := LZ_PATTERN.match(line):
            lz.append(float(m['value']))
            lz_loop.append(int(m['loop'] == 'inner'))
        elif line.startswith(Z_DIFF_LINE_START):
            z_diff.append(float(line[Z_DIFF_VALUE_POSITION:]))
        elif line.startswith(SIGMA_DIFF_LINE_START):
            sigma_diff.append(float(line[SIGMA_DIFF_VALUE_POSITION:]))
        elif line.startswith('INFO:__main__:Resulting betas:'):
            global done
            done = True
        elif line.startswith(TARGET_BETA_START):
            global target_beta
            print(f'Target beta: {line[TARGET_BETA_POSITION:]}')
            target_beta = np.array(json.loads(line[TARGET_BETA_POSITION:]))
        else:
            m = BETA_PATTERN.match(line)
            if m:
                variables = m.groupdict()
                new_beta = json.loads(variables['array'])

                institution_number = int(variables['institution_number'])

                for idx, b in enumerate(new_beta):
                    key = f'{institution_number}_{idx}'

                    if key not in beta.keys():
                        beta[key] = []

                    beta[key].append(b)

                # print(f'Beta:\n{json.dumps(beta)}')
                global beta_df
                beta_df = create_beta_df(beta, target_beta)
                #
                # print(f'Beta df:\n{beta_df.to_json()}')

                new_mae = compute_mae()

                mae.append(new_mae)


#
def create_beta_df(beta: dict, target):
    all_values = []
    all_targets = []
    all_names = []

    # targets
    for i in range(target.shape[0]):
        all_values.append(target[i])
        all_names.append(i)
        all_targets.append(True)

    # New values
    for idx, (k, v) in enumerate(sorted(beta.items())):
        all_values.append(v[-1])
        all_targets.append(False)
        all_names.append(idx)

    return pd.DataFrame({'name': all_names, 'value': all_values, 'target': all_targets})


#
#
# def compute_mse():
#     sum_of_errors = 0
#     num_items = 0
#     for institution, values in beta.items():
#         last_value = values[-1]
#         sum_of_errors += np.square(target_beta[institution] - last_value).sum()
#         num_items += len(values)
#
#     return sum_of_errors / num_items


def compute_mae():
    num_items = 0

    sum_of_errors = 0
    for beta_key, target_index in zip(sorted(beta.keys()), range(target_beta.shape[0])):
        last_value = beta[beta_key][-1]
        sum_of_errors += np.abs(last_value - target_beta[target_index])
        num_items += 1

    #
    # for institution, values in beta.items():
    #     last_value = values[-1]
    #     sum_of_errors += np.abs(target_beta[institution] - last_value).sum()
    #     num_items += len(values)

    return sum_of_errors / num_items


if __name__ == '__main__':
    app.run_server(debug=True)
