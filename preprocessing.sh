#!/bin/bash

BLUE=$'\e[1;34m'
END=$'\e[0m'
printf "\n%s\n\n""${BLUE}Creating virtual environment${END}"
pip3 install virtualenv --user
python3 -m venv ./.virtualenv

printf "\n\n%s\n" "${BLUE}Running virtual environment${END}"
chmod u+x ./.virtualenv/bin/activate
source ./.virtualenv/bin/activate py35


printf "\n%s\n" "${BLUE}Installing prerequisites${END}"
pip3 install -r requirements.txt

printf "\n%s\n" "${BLUE}Running application${END}"

INPUT_FILE=$1
echo $INPUT_FILE
COLS="song_id artist_id artist_mbid artist_familiarity artist_hotttnesss"

printf "\n%s\n" "${BLUE}Removing Columns${END}"
python3 application/preprocessing/remove_columns.py $INPUT_FILE .tmp.pkl $COLS 

printf "\n%s\n" "${BLUE}Partitioning Data${END}"
python3 application/preprocessing/partition.py .tmp.pkl data 0.9 

printf "\n%s\n" "${BLUE}Diagnostics Training Set${END}"
python3 application/preprocessing/investigate.py data/train.pkl data/train_label.pkl data 

printf "\n%s\n" "${BLUE}Diagnostics Test Set${END}"
python3 application/preprocessing/investigate.py data/test.pkl data/test_label.pkl data 


rm .tmp.pkl

deactivate

