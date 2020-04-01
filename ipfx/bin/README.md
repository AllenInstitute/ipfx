Running scripts
===============

Run ephys feature extraction pipeline from the command line
-----------------------------------------------------------

The executable to run pipeline:

```bash
    python run_pipeline.py --input_json <path/to/input.json> --output_json <path/to/output.json>
```

where the input_json file includes input parameters and the output_json file will store the output 

For details on the format of the input see ipfx._schemas.py

If using the default stimulus_ontology and qc_criteria (see ipfx/defaults) the pipeline can be run
with the convenience script below:

```bash
    python run_pipeline_from_nwb_file.py <path/to/input_nwb_file> <output_dir>
```
by directly specifying the name of the input nwb file and output directory as command line arguments.

