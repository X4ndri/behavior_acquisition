#!/bin/bash

/home/ahmad/miniconda3/envs/daq2/bin/python  /home/ahmad/projects/behavior/cammymod/cammy/cli.py run --camera-options  /home/ahmad/projects/behavior/cammy_mod/camera_options.toml --save-engine raw --record --display-downsample 1 --prefix Lucid

read -p "Press any key to exit..."
python 