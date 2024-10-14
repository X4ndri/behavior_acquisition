#!/usr/bin/env bash

# Set the desired terminal window title
new_title="runningScript"
# Modify the terminal window title using the escape sequence
echo -ne "\033]0;$new_title\007"

# Open htop in a new terminal window and set its position and size
xterm -e "htop" &
sleep 1 # Wait for the terminal window to open

# Align the htop terminal to the right side of the screen
# Assuming the screen width is 1920 pixels and window width is 800 pixels
htop_x=$((1920 - 800))  # Calculate X position for the right side
wmctrl -r 'htop' -e 0,$htop_x,0,800,130  # Position window on the right side

# run daq
export PATH="/home/ahmad/miniconda3/bin:$PATH"
source /home/ahmad/miniconda3/etc/profile.d/conda.sh
conda activate daq

# Align the 'runningScript' terminal to the right side of the screen
# Assuming the window height is 800 pixels
runningScript_x=$((1920 - 800))  # Adjust this based on terminal width
wmctrl -r 'runningScript' -e 0,$runningScript_x,155,800,800  # Position window on the right side

# Bring the daq terminal into focus
wmctrl -a 'runningScript'

# Function to close htop window
close_htop() {
    echo "done"
    sleep 0.1
    wmctrl -c "htop"
}

# Trap the EXIT signal to close htop window before exiting
trap close_htop EXIT

python /home/ahmad/projects/behavior/flir_multicam/flir_multicam.py 1
