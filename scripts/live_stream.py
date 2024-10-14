import os
import re
import time
import threading
import sys
import PySpin
import yaml
import ruamel.yaml
from pathlib import Path
from numpy import mean, std, diff, round, argmax, histogram, arange, append
import numpy as np
import termplotlib as tpl
from colorama import just_fix_windows_console
from pynput import keyboard
from contextlib import redirect_stdout
import sys
from stack import stack_frames
import matplotlib.pyplot as plt
import cv2

# Version for general use
def read_config(configname):
    """
    Reads structured config file
    """
    ruamelFile = ruamel.yaml.YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                cfg = ruamelFile.load(f)
        except Exception as err:
            if err.args[2] == "could not determine a constructor for the tag '!!python/tuple'":
                with open(path, 'r') as ymlfile:
                    cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                    # write_config(configname, cfg)
    else:
        raise FileNotFoundError(
            "Config file is not found. Please make sure that the file exists and/or there are no unnecessary spaces in the path of the config file!")
    return (cfg)


# Change cwd to script folder
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Read cfg yaml file
cfg = read_config('/home/ahmad/projects/behavior/flir_multicam/params.yaml')
num_images = cfg['num_images']
exp_time = cfg['exp_time']
# ignore the framerate parameter form the params.yml and set it to always be software @30Hz
framerate = 30
trigger_line = cfg['trigger_line']
bin_val = int(1)  # bin mode (WIP)
gain_val = cfg['gain']

# Thread process for saving images. This is super important, as the writing process takes time inline,
# so offloading it to separate CPU threads allows continuation of image capture
class ThreadShow(threading.Thread):
    def __init__(self, data):
        threading.Thread.__init__(self)
        self.data = data
    def run(self):
        cv2.imshow("cam", self.data.GetNDArray())
        if cv2.waitKey(5) & 0xFF == ord('q'):
            return False
        self.data.Release(0)



# Capturing is also threaded, to increase performance
class ThreadCapture(threading.Thread):
    def __init__(self, cam, camnum, nodemap, stop_event):
        threading.Thread.__init__(self)
        self.cam = cam
        self.camnum = camnum
        self.stop_event = stop_event


    def run(self):
   
        if framerate != 'hardware':
            nodemap = self.cam.GetNodeMap()

        if self.camnum == 0:
            primary = 1
        else:
            primary = 0   
        i = 0
        grabTimeout = 1000 #ms
        # for i in range(num_images):
        while i < num_images:
            fstart = time.perf_counter_ns()
            try:
                #  Retrieve next received image
                if framerate == 'hardware':
                    if i == 0:
                        image_result = self.cam.GetNextImage()
                        cv2.imshow(f"cam {self.camnum}", im)
                        if cv2.waitKey(5) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break

                else:
                    node_softwaretrigger_cmd = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
                    if not PySpin.IsAvailable(node_softwaretrigger_cmd) or not PySpin.IsWritable(
                            node_softwaretrigger_cmd):
                        print('Unable to execute trigger. Aborting...')
                        return False
                    node_softwaretrigger_cmd.Execute()
                    image_result = self.cam.GetNextImage()
                    im = image_result.GetNDArray()
                    # Display the image in a window
                    cv2.imshow(f"cam {self.camnum}", im)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
            except PySpin.SpinnakerException as ex:
                pass

        self.cam.EndAcquisition()

        


def config_and_acquire(camlist):
    global stop_event
    stop_event = threading.Event()

    thread = []
    # append keyboard listener thread
    kb_listener = keyboard.Listener(on_press=on_press, on_release=None)
    kb_listener.start()
    print(camlist)
    for i, cam in enumerate(camlist):
        print(i)
        cam.Init()
        configure_cam(cam, i)
        nodemap = cam.GetNodeMap()
        cam.BeginAcquisition()
        thread.append(ThreadCapture(cam, i, nodemap, stop_event))
        thread[i].start()
    # after all cameras threads, append the listener thread
    thread.append(kb_listener)



    if framerate == 'hardware':
        print('*** WAITING FOR FIRST TRIGGER... ***\n')
    for t in thread:
        t.join()

    for i, cam in enumerate(camlist):
        reset_trigger(cam)
        cam.DeInit()


# Config camera params, but don't begin acquisition
def config_and_return(camlist):
    for i, cam in enumerate(camlist):
        cam.Init()
        configure_cam(cam, i)

    for i, cam in enumerate(camlist):
        reset_trigger(cam)
        cam.DeInit()

# Trigger reset
def reset_trigger(cam):
    nodemap = cam.GetNodeMap()
    try:
        result = True
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            print('Unable to disable trigger mode 630 (node retrieval). Aborting...')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

    except PySpin.SpinnakerException as ex:
        print('Error (663): %s' % ex)
        result = False

    return result
# add a keyboard listener thread
def on_press(key):
    try:
        # Check if the pressed key is 'q'
        if key.char == 'q':
            stop_event.set()
            print('\n')
            print("'q' press detected, stopping...")
            return False
    except AttributeError:
        # Ignore special keys like 'Shift', 'Ctrl', etc.
        pass
# def on_release(key):
#     # Stop the listener when 'q' key is released
#     if key == keyboard.Key.esc:
#         return False

# Main writing loop
def main():

    result = True
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    print('Number of cameras detected: %d' % num_cameras)
    cam_live_view = int(sys.argv[1])
    print(f'viewing camera {cam_live_view}')

    # cam_live_view = int(input("which camera to view? index starts at 0.. "))
    if num_cameras == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print('Not enough cameras! Goodbye.')
        return False

    elif num_cameras > 0:
        config_and_acquire([cam_list[cam_live_view]])


    # Clear cameras and release system instance
    cam_list.Clear()
    system.ReleaseInstance()

    return result



def configure_cam(cam, camnum):

    result = True
    if camnum == 0:
        print('*** CONFIGURING CAMERA(S) ***\n')
    print(f'Configuring cam {camnum}\n\n')
    try:
        nodemap = cam.GetNodeMap()
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            print('Unable to disable trigger mode 129 (node retrieval). Aborting...')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
            print('Unable to get trigger source 163 (node retrieval). Aborting...')
            return False

        # Set primary camera trigger source to cfg['trigger_line'] (hardware trigger)
        if framerate == 'hardware':
            node_trigger_source_set = node_trigger_source.GetEntryByName(trigger_line)
            if camnum == 0:
                print('Trigger source set to hardware...\n')
        else:
            node_trigger_source_set = node_trigger_source.GetEntryByName('Software')
            if camnum == 0:
                print('Trigger source set to software, framerate = %i...\n' % framerate)

        if not PySpin.IsAvailable(node_trigger_source_set) or not PySpin.IsReadable(
                node_trigger_source_set):
            print('Unable to set trigger source (enum entry retrieval). Aborting...')
            return False

        node_trigger_source.SetIntValue(node_trigger_source_set.GetValue())
        node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')

        if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
            print('Unable to enable trigger mode (enum entry retrieval). Aborting...')
            return False

 
        node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())

        # Set acquisition mode to continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        # Retrieve Stream Parameters device nodemap
        s_node_map = cam.GetTLStreamNodeMap()

        # Retrieve Buffer Handling Mode Information
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
            print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
            return False

        handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not PySpin.IsAvailable(handling_mode_entry) or not PySpin.IsReadable(handling_mode_entry):
            print('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')
            return False

        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
        if not PySpin.IsAvailable(stream_buffer_count_mode) or not PySpin.IsWritable(stream_buffer_count_mode):
            print('Unable to set Buffer Count Mode (node retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
        if not PySpin.IsAvailable(stream_buffer_count_mode_manual) or not PySpin.IsReadable(
                stream_buffer_count_mode_manual):
            print('Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())

        # Retrieve and modify Stream Buffer Count
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
        if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
            print('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')
            return False

        # Set new buffer value to the max
        max_buffer_count = buffer_count.GetMax()
        if camnum==0:
            print(f"Setting buffer count to: {max_buffer_count}")
        # this line seems to halt the cameras and breaks the cam.BeginAcquisition() method
        buffer_count.SetValue(max_buffer_count)

        max_packet_size = cam.DiscoverMaxPacketSize()
        # Retrieve and modify resolution (WIP)
        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):
            width_to_set = int(1440 / bin_val)
            node_width.SetValue(width_to_set)
            if camnum == 0:
                print('Width set to %i...' % node_width.GetValue())
        else:
            if camnum == 0:
                print('Width not available, width is %i...' % node_width.GetValue())
        
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if PySpin.IsAvailable(node_height) and PySpin.IsWritable(node_height):
            height_to_set = int(1080 / bin_val)
            node_height.SetValue(height_to_set)
            if camnum == 0:
                print('Height set to %i...' % node_height.GetValue())
        else:
            if camnum == 0:
                print('Width not available, height is %i...' % node_height.GetValue())

        # Access trigger overlap info
        node_trigger_overlap = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerOverlap'))
        if not PySpin.IsAvailable(node_trigger_overlap) or not PySpin.IsWritable(node_trigger_overlap):
            print('Unable to set trigger overlap to "Read Out". Aborting...')
            return False

        # Retrieve enumeration for trigger overlap Read Out
        if framerate == 'hardware':
            node_trigger_overlap_ro = node_trigger_overlap.GetEntryByName('ReadOut')
        else:
            node_trigger_overlap_ro = node_trigger_overlap.GetEntryByName('Off')

        if not PySpin.IsAvailable(node_trigger_overlap_ro) or not PySpin.IsReadable(
                node_trigger_overlap_ro):
            print('Unable to set trigger overlap (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from enumeration
        trigger_overlap_ro = node_trigger_overlap_ro.GetValue()

        # Set trigger overlap using retrieved integer from enumeration
        node_trigger_overlap.SetIntValue(trigger_overlap_ro)

        # Access exposure auto info
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if not PySpin.IsAvailable(node_exposure_auto) or not PySpin.IsWritable(node_exposure_auto):
            print('Unable to get exposure auto. Aborting...')
            return False

        # Retrieve enumeration for trigger overlap Read Out
        node_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_exposure_auto_off) or not PySpin.IsReadable(
                node_exposure_auto_off):
            print('Unable to get exposure auto "Off" (entry retrieval). Aborting...')
            return False
        
        if gain_val == 'auto':
            # Set Gain to a fixed value
            if PySpin.IsWritable(nodemap.GetNode("GainAuto")):
                gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
                if gain_auto and gain_auto.GetEntryByName("Once"):
                    # Set GainAuto to Off to disable automatic gain control
                    gain_auto.SetIntValue(gain_auto.GetEntryByName("Once").GetValue())
                    print("AutoGain set to Off")
                else:
                    print("Unable to set GainAuto to Off")
        else:
            gain_node = nodemap.GetNode("Gain")
            if PySpin.IsWritable(gain_node):
                gain = PySpin.CFloatPtr(gain_node)
                if gain:
                    gain.SetValue(gain_val)
                    print(f"Gain set to {gain_val}")
                else:
                    print("Unable to access Gain node")
            else:
                print("Gain node is not writable")

        # Access exposure info
        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if not PySpin.IsAvailable(node_exposure_time) or not PySpin.IsWritable(node_exposure_time):
            print('Unable to get exposure time. Aborting...')
            return False

        # Set exposure float value
        node_exposure_time.SetValue(exp_time * 1000000)
        if camnum == 0:
            print('Exposure time set to ' + str(exp_time * 1000) + 'ms...')

    # General exception
    except PySpin.SpinnakerException as ex:
        print('Error (237): %s' % ex)
        return False

    return result


if __name__ == '__main__':
    main()
