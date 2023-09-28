from sksurv.functions import StepFunction
import numpy as np

DEFAULT_DECIMAL = 4


def compare_stepfunctions(a: StepFunction, b: StepFunction, decimal=DEFAULT_DECIMAL):
    print('Comparing x')
    np.testing.assert_almost_equal(a.x, b.x, decimal=decimal)

    print('Comparing y')
    for x in a.x:
        print(x)
        print(f'x: {x}, a(x): {a(x)}, b(x): {b(x)}')
        np.testing.assert_almost_equal(a(x), b(x), decimal=decimal)
