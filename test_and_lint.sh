echo
echo "================================== pytest =================================="
echo
pytest
echo
echo "================================== pylint =================================="
echo
pylint application/*.py application/visualization/*.py application/preprocessing/*.py tests/*.py 

