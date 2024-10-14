#!/usr/bin/env bash

# Set the desired terminal window title
new_title="runningScript"
# Modify the terminal window title using the escape sequence
echo -ne "\033]0;$new_title\007"

# Open htop in a new terminal window and set its position and size
xterm -e "htop" &
sleep 1 # Wait for the terminal window to open
wmctrl -r 'htop' -e 0,0,0,800,130

# run daq
export PATH="/home/ahmad/miniconda3/bin:$PATH"
source /home/ahmad/miniconda3/etc/profile.d/conda.sh
conda activate daq

wmctrl -r 'runningScript' -e 0,0,155,800,800
# bring the daq terminal in focus
wmctrl -a 'runningScript'

# Function to close htop window
close_htop() {
    echo "done"
    sleep 0.1
    wmctrl -c "htop"
}

# Trap the EXIT signal to close htop window before exiting
trap close_htop EXIT

python  /home/ahmad/projects/behavior/flir_multicam/flir_multicam.py 1
