#!/bin/bash

SPECIMEN_NAMES_TXT=$1

for p in $(<$SPECIMEN_NAMES_TXT)
do
    echo ${p}
    bash run_patchseq_specimen.sh ${p}

done
