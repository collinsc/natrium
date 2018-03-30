#!/usr/bin/env bash
echo
echo "================================== pytest =================================="
echo
pytest
echo
echo "================================== pylint =================================="
echo
pylint application/*.py tests/*.py