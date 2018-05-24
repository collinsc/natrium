#!/bin/bash

#printf "\nCreating virtual environment\n\n"
#pip3 install virtualenv --user
#python3 -m venv ./.virtualenv
#
#printf "\nRunning virtual environment\n\n"
#chmod u+x ./.virtualenv/bin/activate
#source ./.virtualenv/bin/activate 
#
#
#printf "Installing prerequisites\n\n"
#pip3 install -r requirements.txt

printf "\nRunning application\n\n"

SCRIPT=$1
shift
cd application
python3 -m  $SCRIPT $@

#deactivate

