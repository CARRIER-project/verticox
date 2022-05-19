import numpy as np

def test_group_samples_at_risk_numbers_descend():
    event_times = np.array([4,7, 7, 6, 7,23, 2, 4, ])
    # Testing if the resulting list descends in numbers
    previous_length = len(event_times) + 1

    for t in sorted(Rt.keys()):
        length = len(Rt[t])

        assert length < previous_length

        previous_length = length


test_group_samples_at_risk_numbers_descend()