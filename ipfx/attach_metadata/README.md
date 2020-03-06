Attach Metadata
===============

A utility for dispatching metadata (lightweight pieces of ancillary information about an experiment) to appropriate sinks. Currently, we support 

- out-of-place NWB2 appending
- writing to a yaml, for eventual upload to DANDI

To invoke from the command line, run:

```bash
    python -m ipfx.attach_metadata --input_json <path/to/input.json> --output_json <path/to/output.json>
```
For details on the formats of the input and output jsons, see `_schemas.py`