import ipfx.sweep_props as sp


def test_count_sweep_states():

    sweep_states = [
        {
            "reasons": [],
            "sweep_number": 14,
            "passed": True
        },
        {
            "reasons": [
                "slow noise: 0.502 above threshold: 0.500"
            ],
            "sweep_number": 15,
            "passed": False
        },
        {
            "reasons": [],
            "sweep_number": 16,
            "passed": True
        },
        {
            "reasons": [],
            "sweep_number": 17,
            "passed": True
        },
        {
            "reasons": [
                "slow noise: 1.208 above threshold: 0.500"
            ],
            "sweep_number": 18,
            "passed": False
        },

    ]

    num_passed_sweeps, num_sweeps = sp.count_sweep_states(sweep_states)
    assert num_passed_sweeps == 3
    assert num_sweeps == 5
