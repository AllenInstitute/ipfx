CELL=$1
CELL_DIR=/local1/ephys/patchseq/specimens/$CELL
INPUT_JSON=$CELL_DIR/pipeline_input.json
OUTPUT_JSON=$CELL_DIR/pipeline_output.json
LOG_FILE=$CELL_DIR/log.txt

mkdir -p $CELL_DIR
python generate_pipeline_input_patchseq.py $CELL $CELL_DIR
python run_pipeline.py --input_json $INPUT_JSON --output_json $OUTPUT_JSON --log_level DEBUG |& tee $LOG_FILE
