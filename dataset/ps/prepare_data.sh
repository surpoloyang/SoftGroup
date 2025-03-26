#!/bin/bash
python prepare_data_inst.py --verbose
python downsample.py --verbose
python prepare_data_inst_gttxt.py
