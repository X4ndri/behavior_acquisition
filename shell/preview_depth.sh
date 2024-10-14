#!/bin/bash

/home/ahmad/miniconda3/envs/daq2/bin/python  /home/ahmad/projects/behavior/cammy_mod/cammy/cli.py run --camera-options  /home/ahmad/projects/behavior/cammy_mod/camera_options.toml --display-downsample 1 --prefix Lucid

read -p "Press any key to exit..."
python 