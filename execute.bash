#!/bin/bash

printf "\nCreating virtual environment\n\n"
pip3 install virtualenv --user
python3 -m venv ./.virtualenv

printf "\nRunning virtual environment\n\n"
chmod u+x ./.virtualenv/bin/activate
source ./.virtualenv/bin/activate py35


printf "Installing prerequisites\n\n"
pip3 install -r requirements.txt

cd application
printf "\nRunning application\n\n"

SCRIPT=$1
shift
python3 $SCRIPT $@

deactivate

