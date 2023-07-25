import clize
import pandas as pd

from test_constants import CONVERGENCE_PRECISION
from verticox.node_manager import LocalNodeManager


def run_locally(data, event_times_column, event_happened_column):
    df = pd.read_parquet(data)

    print(df)

    node_manager = LocalNodeManager(df, event_times_column, event_happened_column,
                                    {'convergence_precision': CONVERGENCE_PRECISION})
    node_manager.start_nodes()
    node_manager.fit()

    print(node_manager.betas)


if __name__ == '__main__':
    clize.run(run_locally)
