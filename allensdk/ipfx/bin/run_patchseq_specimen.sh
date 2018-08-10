INPUT_NWB_FILE_FULL_PATH=$1

INPUT_NWB_FILE=${INPUT_NWB_FILE_FULL_PATH##*/}
SPECIMEN_NAME=${INPUT_NWB_FILE%.*}

OUTPUT_DIR=/allen/aibs/technology/sergeyg/ephys_pipeline/patch_seq_recent_experiments/$SPECIMEN_NAME
INPUT_JSON=$OUTPUT_DIR/pipeline_input.json
OUTPUT_JSON=$OUTPUT_DIR/pipeline_output.json
LOG_FILE=$OUTPUT_DIR/log.txt

mkdir -p $OUTPUT_DIR

python generate_patchseq_pipeline_input.py "${INPUT_NWB_FILE_FULL_PATH}" "${OUTPUT_DIR}"
python run_pipeline.py --input_json $INPUT_JSON --output_json $OUTPUT_JSON --log_level DEBUG |& tee $LOG_FILE
