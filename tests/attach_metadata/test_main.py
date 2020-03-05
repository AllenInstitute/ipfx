"""Tests for the utilities in attach_metadata.__main__

See test_cli for integration tests of
    run_attach_metadata()
    main()

"""
from ipfx.attach_metadata import __main__ as attach
from ipfx.attach_metadata.sink import MetadataSink


def test_configure_sinks():

    class SinkTest:
        def register_targets(self, targets):
            self.targets = targets

    class SinkA(SinkTest):
        def __init__(self, a):
            self.a = a
        
    class SinkB(SinkTest):
        def __init__(self, b):
            self.b = b
    
    sink_specs = [
        {
            "name": "a1",
            "kind": "SinkA",
            "config": {"a": 1},
            "targets": [1]
        },
        {
            "name": "a2",
            "kind": "SinkA",
            "config": {"a": 2},
            "targets": [2]
        },
        {
            "name": "b1",
            "kind": "SinkB",
            "config": {"b": 1},
            "targets": [3]
        }
    ]
    sink_kinds = {"SinkA": SinkA, "SinkB": SinkB}
    obtained = attach.configure_sinks(sink_specs, sink_kinds)

    assert obtained["b1"].targets[0] == 3
    assert obtained["a1"].a == 1 
    assert obtained["a2"].a == 2


def test_attach_metadata():

    class SinkTest:
        def __init__(self):
            self.registered = {}

        def register(self, name, value, sweep_number):
            self.registered[(name, sweep_number)] = value

    sinks = {"fizz": SinkTest(), "buzz": SinkTest()}

    attach.attach_metadata(
        sinks,
        [
            {
                "name": "a",
                "value": 1,
                "sinks": ["fizz", "buzz"],
                "sweep_number": 12
            },
            {
                "name": "b",
                "value": 2,
                "sinks": ["buzz"]
            }
        ]
    )

    assert sinks["fizz"].registered[("a", 12)] == 1 
    assert sinks["buzz"].registered[("a", 12)] == 1 
    assert sinks["buzz"].registered[("b", None)] == 2 

