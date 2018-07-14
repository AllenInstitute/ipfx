#PBS -q celltypes
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=1
#PBS -N cell_job
#PBS -r n
#PBS -j oe
#PBS -o /allen/aibs/technology/sergeyg/ephys_pipeline/job.out
#PBS -m a
#PBS -l mem=10gb

cd $PBS_O_WORKDIR

echo $SPECIMEN_ID

CELL_DIR=/allen/aibs/technology/sergeyg/ephys_pipeline/specimen_$SPECIMEN_ID
INPUT_JSON=$CELL_DIR/pipeline_input.json
OUTPUT_JSON=$CELL_DIR/pipeline_output.json
LOG_FILE=$CELL_DIR/log_$SPECIMEN_ID.txt

mkdir -p $CELL_DIR

source activate ipfx

python generate_pipeline_input.py $SPECIMEN_ID $CELL_DIR
python run_pipeline.py --input_json $INPUT_JSON --output_json $OUTPUT_JSON --log_level DEBUG |& tee $LOG_FILE
python validate_experiment.py $INPUT_JSON $OUTPUT_JSON --log_level DEBUG |& tee -a $LOG_FILE
