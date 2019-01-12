#!/bin/bash
cd python-mnist
./get_data.sh
cd ..
export PYTHONPATH=$PYTHONPATH:python-mnist
