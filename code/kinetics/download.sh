#!/usr/bin/env bash

DATASET="kinetics400"
DATA_DIR="../../../data/${DATASET}/annotations"

bash download_annotations.sh ${DATASET}
python select_labels.py ${DATA_DIR}/kinetics_train.csv
python select_labels.py ${DATA_DIR}/kinetics_val.csv
python select_labels.py ${DATA_DIR}/kinetics_test.csv
bash download_videos.sh ${DATASET}
bash rename_classnames.sh ${DATASET}
bash remove_corrupt_files.sh
bash generate_videos_filelist.sh ${DATASET}