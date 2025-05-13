#!/bin/bash
python prepare_data_inst.py --align
python downsample.py
python prepare_data_inst_gttxt.py
