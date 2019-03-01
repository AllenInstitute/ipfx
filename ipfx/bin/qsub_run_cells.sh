#!/bin/bash

EPHYS_PIPELINE_DIR=/allen/aibs/technology/sergeyg/ephys_pipeline

SPECIMEN_IDS_TXT=$EPHYS_PIPELINE_DIR/specimen_ids.txt

for p in $(<$SPECIMEN_IDS_TXT)
do
    echo ${p}
    JOB_NAME=cell_$p
    JOB_OUTPUT_FILE=$EPHYS_PIPELINE_DIR/specimen_$p/cell_$p.out

    qsub -N $JOB_NAME -o $JOB_OUTPUT_FILE -v SPECIMEN_ID="${p}" run_and_validate_specimen_queue.sh

done
