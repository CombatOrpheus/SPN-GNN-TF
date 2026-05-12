#!/bin/bash
export PYTHONPATH=$PYTHONPATH:src
rm -rf tests/*.cache*
~/.pixi/bin/pixi run python -m unittest tests/test_baseline_models.py
rm -rf tests/*.cache*
~/.pixi/bin/pixi run python -m unittest tests/test_integration.py
