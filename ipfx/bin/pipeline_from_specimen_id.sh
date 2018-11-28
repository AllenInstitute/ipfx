OUTPUT_DIR=/local1/ephys/ivscc/testspecimens
SPECIMEN_ID=$1

CELL_NAME=$SPECIMEN_ID
CELL_DIR=$OUTPUT_DIR/$CELL_NAME
INPUT_JSON=$CELL_DIR/pipeline_input.json
OUTPUT_JSON=$CELL_DIR/pipeline_output.json
LOG_FILE=$CELL_DIR/log.txt

echo $CELL_DIR

mkdir -p $CELL_DIR

python generate_pipeline_input.py --specimen_id $SPECIMEN_ID --cell_dir $CELL_DIR
python run_pipeline.py --input_json $INPUT_JSON --output_json $OUTPUT_JSON --log_level DEBUG |& tee $LOG_FILE
