#!/bin/sh

OUTPUT_DIR=/local1/ephys/tsts

INPUT_NWB_FILE_FULL_PATH=$1
INPUT_NWB_FILE=${INPUT_NWB_FILE_FULL_PATH##*/}
CELL_NAME=${INPUT_NWB_FILE%.*}

CELL_DIR=$OUTPUT_DIR/$CELL_NAME
INPUT_JSON=$CELL_DIR/pipeline_input.json
OUTPUT_JSON=$CELL_DIR/pipeline_output.json
LOG_FILE=$CELL_DIR/log.txt

mkdir -p $CELL_DIR

python generate_pipeline_input.py --input_nwb_file "${INPUT_NWB_FILE_FULL_PATH}" --cell_dir "${CELL_DIR}"
python run_pipeline.py --input_json $INPUT_JSON --output_json $OUTPUT_JSON --log_level DEBUG |& tee $LOG_FILE
