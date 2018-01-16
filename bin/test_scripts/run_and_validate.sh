CELL=$1
INPUT_JSON=specimen_$CELL/pipeline_input.json
OUTPUT_JSON=specimen_$CELL/pipeline_output.json

python test_scripts/generate_pipeline_input.py $CELL
python run_pipeline.py --input_json $INPUT_JSON --output_json $OUTPUT_JSON --log_level DEBUG
python test_scripts/validate_experiment.py $INPUT_JSON $OUTPUT_JSON