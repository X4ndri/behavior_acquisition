#!/bin/bash

#wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz
wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz

sudo /home/ahmad/miniconda3/envs/daq2/bin/python  /home/ahmad/projects/behavior/scripts/move_to_nas.py


read -p "Press any key to exit..."
python 