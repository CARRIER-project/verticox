from sksurv.functions import StepFunction
import numpy as np

def compare_stepfunctions(a: StepFunction, b: StepFunction):
    print('Comparing x')
    np.testing.assert_almost_equal(a.x, b.x)

    print('Comparing y')
    np.testing.assert_almost_equal(a(a.x), b(a.x))