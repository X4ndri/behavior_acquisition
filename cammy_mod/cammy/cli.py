import click
import numpy as np
import toml
import logging
import sys
import os
import time
import cv2
import yaml
from pathlib import Path
import subprocess


cwd = Path(__file__)
pwd = cwd.parent.parent.parent

FLIR_PARAMS_PATH = pwd.joinpath('flir_multicam/params.yaml')
FLIR_BASH_PATH = pwd.joinpath("shell/start_side_cameras_acqusition.sh")
DEPTH_SAVE_DIR = "/mnt/data_sp/depth_behavior/"
SIDE_CAM_SAVE_DIR = "/mnt/data_nvme/side_camera_behavior/"


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s]:%(levelname)s:%(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


from typing import Optional, Iterable
from cammy.util import (
    get_all_camera_ids,
    intensity_to_rgba,
    get_queues,
    initialize_cameras,
    get_output_format,
    get_pixel_format_bit_depth,
    mpl_to_cv2_colormap,
    check_counters_equal,
)
from cammy.record.video import FfmpegVideoRecorder, RawVideoRecorder, FrameWriter
from cammy.camera.spoof import SpoofCamera

@click.group()
def cli():
    pass


slider_defaults_min = {
    "default_value": 1800,
    "min_value": 0,
    "max_value": 5000,
}

slider_defaults_max = {
    "default_value": 2200,
    "min_value": 0,
    "max_value": 5000,
}
colormap_default = "gray"
gui_ncols = 3  # number of cols before we start new row
# for labeling videos
font = cv2.FONT_HERSHEY_SIMPLEX
white = (255, 255, 255)
txt_pos = (25, 25)


@cli.command(name="run", context_settings={'show_default': True})
@click.option("--interface", type=click.Choice(["aravis", "fake_custom", "all"]), default="all")
@click.option("--buffer-size", "-b", type=int, default=5, help="Buffer size")
@click.option("--n-fake-cameras", type=int, default=1)
@click.option("--record", is_flag=True, help="Save frames to disk")
@click.option("--jumbo-frames", default=True, type=bool, help="Turn on jumbo frames (GigE only)")
@click.option(
    "--save-engine",
    type=click.Choice(["ffmpeg", "raw", "frames"]),
    default="ffmpeg",
    help="Save raw frames or compressed frames using ffmpeg",
)
@click.option(
    "--display-downsample",
    type=int,
    default=1,
    help="Downsample frames for display (full data is saved)",
)
@click.option("--display-colormap", type=str, default="turbo", help="Look-up-table")
@click.option("--duration", type=float, default=0, help="Run for N minutes")
@click.option(
    "--camera-options",
    type=click.Path(resolve_path=True),
    default="camera_options.toml",
    help="TOML file with camera options",
)

@click.option(
    "--prefix",
    type=str,
    default=None,
    help="Only uses cameras whose IDs start with this string (None to use all IDs)"
)
# fmt: on
def simple_preview(
    interface: str,
    buffer_size: int,
    n_fake_cameras: int,
    camera_options: Optional[str],
    record: bool,
    jumbo_frames: bool,
    save_engine: str,
    display_downsample: int,
    display_colormap: Optional[str],
    duration: float,
    prefix: Optional[str],
):
    
    cli_params = locals()

    import dearpygui.dearpygui as dpg
    import cv2
    import datetime

    basedir = os.path.dirname(os.path.abspath(__file__))
    # save_base_dir = "/mnt/data_nvme/intercam_nvme/"
    save_base_dir = "/mnt/data_nvme/intercam_nvme/"
    if display_colormap is None:
        display_colormap = mpl_to_cv2_colormap(colormap_default)
    else:
        display_colormap = mpl_to_cv2_colormap(display_colormap)

    if (camera_options is not None) and os.path.exists(camera_options):
        logging.info(f"Loading camera options from {camera_options}")
        camera_dct = toml.load(camera_options)
    else:
        camera_dct = {}

    cameras = {}
    ids = get_all_camera_ids(interface, n_cams=n_fake_cameras, prefix=prefix)

    cameras = initialize_cameras(
        ids,
        camera_dct,
        jumbo_frames=jumbo_frames,
        buffer_size=buffer_size,
    )
    del cameras
    time.sleep(2)

    cameras_metadata = {}
    bit_depth = {}
    spoof_cameras = {}
    trigger_pins = []
    cameras = initialize_cameras(
        ids,
        camera_dct,
        jumbo_frames=jumbo_frames,
        buffer_size=buffer_size,
    )
    for i, (k, v) in enumerate(cameras.items()):
        feature_dct = v.get_all_features()
        feature_dct = dict(sorted(feature_dct.items()))

        # make sure we set so we know how to decode frame buffers
        v._pixel_format = feature_dct["PixelFormat"]
        _bit_depth, _spoof_ims = get_pixel_format_bit_depth(feature_dct["PixelFormat"])
        bit_depth[k] = _bit_depth
        cameras_metadata[k] = feature_dct
        for _spoof_name, _spoof_bit_depth in _spoof_ims.items():
            new_id = f"{k}-{_spoof_name}"
            spoof_cameras[new_id] = SpoofCamera(id=new_id)
            spoof_cameras[new_id]._width = v._width
            spoof_cameras[new_id]._height = v._height
            bit_depth[new_id] = _spoof_bit_depth
            v._spoof_cameras.append(spoof_cameras[new_id])

    # merge in spoof cameras, should be no key collisions
    cameras = cameras | spoof_cameras
    ids = ids | {_id: "spoof" for _id in spoof_cameras.keys()}

    cameras = dict(sorted(cameras.items()))
    ids = dict(sorted(ids.items()))

    recorders = []
    write_dtype = {}
    if record:
        # from parameters construct single names...
        use_queues = get_queues(list(ids.keys()))
        metadata_path = os.path.join(basedir, "metadata.toml")
        show_fields = toml.load(metadata_path)["show_fields"]
        init_timestamp = datetime.datetime.now()

        recording_metadata = {
            "codec": "raw",
            "start_time": init_timestamp.isoformat(),
            "cameras": ids,
            "bit_depth": bit_depth,
            "camera_metadata": cameras_metadata,
            "cli_parameters": cli_params,
        }

        write_dtype, codec = get_output_format(save_engine, bit_depth)
        recording_metadata["codec"] = codec
        recording_metadata["pixel_format"] = write_dtype

        init_timestamp_str = init_timestamp.strftime("%Y%m%d%H%M%S-%f")
        dpg.create_context()
        # save_path = os.path.abspath(f"{info}_{init_timestamp_str}")
        # https://github.com/hoffstadt/DearPyGui/issues/1380
        with dpg.font_registry():
            # Download font here: https://fonts.google.com/specimen/Open+Sans
            font_path = os.path.join(
                basedir, "../assets", "OpenSans-VariableFont_wdth,wght.ttf"
            )
            default_font_large = dpg.add_font(font_path, 24 * 2, tag="ttf-font-large")
            default_font_small = dpg.add_font(font_path, 20 * 2, tag="ttf-font-small")

        settings_tags = {}
        settings_vals = {}

        with dpg.window(width=500, height=300, no_resize=True, tag="settings"):
            for k, v in show_fields.items():
                settings_tags[k] = dpg.add_input_text(default_value=v, label=k)
            dpg.add_spacer(height=5)
            dpg.add_spacing(count=5)

            def button_callback(sender, app_data):
                for k, v in settings_tags.items():
                    settings_vals[k] = dpg.get_value(v)
                dpg.stop_dearpygui()

            dpg.add_button(label="START EXPERIMENT", callback=button_callback)
            dpg.bind_font(default_font_large)
            dpg.set_global_font_scale(0.5)

        dpg.create_viewport(width=300, height=300, title="Settings")
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("settings", True)
        dpg.start_dearpygui()
        dpg.destroy_context()

        recording_metadata["user_input"] = settings_vals

        info = list(settings_vals.values())
        group_name = info[0]
        mouse_name = info[1]
        injection = info[2]
        dose = info[3]
        # get session parameters info
        info = [str(x) for x in info]
        info = '_'.join(info)
        session_sequence_name = f"{group_name}/{mouse_name}/{injection}/{dose}"
        # prep folder names: side and depth cameras  
        depth_save_path = Path(DEPTH_SAVE_DIR).joinpath(session_sequence_name)
        side_camera_save_path = Path(SIDE_CAM_SAVE_DIR).joinpath(session_sequence_name)
        # make dirs
        os.makedirs(depth_save_path.as_posix(), exist_ok=True)
        os.makedirs(side_camera_save_path.as_posix(), exist_ok=True)
        # make the final child session folders
        depth_files_path = depth_save_path.joinpath(f"{info}_{init_timestamp_str}")
        side_camera_files_path = side_camera_save_path.joinpath(f"{info}_{init_timestamp_str}")
        # make sure no session with the same name exists
        if os.path.exists(depth_files_path):
            raise RuntimeError(f"Directory {depth_files_path} already exists")
        if os.path.exists(side_camera_files_path):
            raise RuntimeError(f"Directory {side_camera_files_path} already exists")
        # put the side cameras files in the side_cameras folder, useful for later
        side_camera_files_path = side_camera_files_path.joinpath('side_cameras')
        os.makedirs(side_camera_files_path.as_posix())
        os.makedirs(depth_files_path.as_posix())
        
        # now we want to send the filename to the flir_multicam script
        # instead restructuring that code, we dump the file name to the
        # flir_multicam config before we call it, so it reads it normally.

        logging.info("Configuring FLIR Parameters...")
        # Read the YAML file
        with open(FLIR_PARAMS_PATH, 'r') as file:
            data = yaml.safe_load(file)
        # change the filepath value
        data['file_path'] = str(side_camera_files_path.as_posix())
        with open(FLIR_PARAMS_PATH, 'w') as file:
            yaml.safe_dump(data, file)
        process = subprocess.Popen(['bash', FLIR_BASH_PATH])

        # done with flir_multicam calling, go back to depth stuff
        # dump in depth metadata
        with open(os.path.join(depth_files_path, "metadata.toml"), "w") as f:
            toml.dump(recording_metadata, f)

        # initiate recorder objects
        for _id, _cam in cameras.items():
            cameras[_id].save_queue = use_queues["storage"][_id]
            timestamp_fields = ["frame_id", "device_timestamp", "system_timestamp"]

            if save_engine == "ffmpeg":
                _recorder = FfmpegVideoRecorder(
                    width=cameras[_id]._width,
                    height=cameras[_id]._height,
                    save_queue=cameras[_id].save_queue,
                    filename=os.path.join(depth_files_path, f"{_id}.mkv"),
                    pixel_format=write_dtype[_id],
                    timestamp_fields=timestamp_fields,
                )
            elif save_engine == "raw":
                _recorder = RawVideoRecorder(
                    save_queue=cameras[_id].save_queue,
                    filename=os.path.join(depth_files_path, f"{_id}.dat"),
                    write_dtype=write_dtype[_id],
                    timestamp_fields=timestamp_fields,
                )
            elif save_engine == 'frames':
                _recorder = FrameWriter(
                    save_queue = cameras[_id].save_queue,
                    timestamp_fields=timestamp_fields,
                    extension = write_dtype[_id],
                    foldername = os.path.join(depth_files_path, f"{_id}")                    
                )

            else:
                raise RuntimeError(
                    f"Did not understanding VideoRecorder option {save_engine}"
                )

            _recorder.daemon = True
            _recorder.start()
            recorders.append(_recorder)
    else:
        show_fields = {}
        use_queues = {}
        depth_files_path = None
        recording_metadata = None


    # start a new context for acquisition
    dpg.create_context()

    # https://github.com/hoffstadt/DearPyGui/issues/1380
    with dpg.font_registry():
        # Download font here: https://fonts.google.com/specimen/Open+Sans
        font_path = os.path.join(
            basedir, "../assets", "OpenSans-VariableFont_wdth,wght.ttf"
        )
        default_font_large = dpg.add_font(font_path, 20 * 2, tag="ttf-font-large")
        default_font_small = dpg.add_font(font_path, 16 * 2, tag="ttf-font-small")

    with dpg.texture_registry(show=False):
        for _id, _cam in cameras.items():
            blank_data = np.zeros(
                (
                    _cam._height // display_downsample,
                    _cam._width // display_downsample,
                    4,
                ),
                dtype="float32",
            )
            dpg.add_raw_texture(
                _cam._width / display_downsample,
                _cam._height / display_downsample,
                blank_data,
                tag=f"texture_{_id}",
                format=dpg.mvFormat_Float_rgba,
            )

    miss_status = {}
    fps_status = {}
    for _id, _cam in cameras.items():
        use_config = {}
        for k, v in camera_dct["display"].items():
            if k in _id:
                use_config = v

        with dpg.window(
            label=f"Camera {_id}",
            tag=f"Camera {_id}",
            no_collapse=True,
            no_scrollbar=True,
        ):
            dpg.add_image(f"texture_{_id}")
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    tag=f"texture_{_id}_min",
                    width=(_cam._width // display_downsample) / 3,
                    **{**slider_defaults_min, **use_config["slider_defaults_min"]},
                )
                dpg.add_slider_float(
                    tag=f"texture_{_id}_max",
                    width=(_cam._width // display_downsample) / 3,
                    **{**slider_defaults_max, **use_config["slider_defaults_max"]},
                )
            miss_status[_id] = dpg.add_text("0 missed frames / 0 total")
            fps_status[_id] = dpg.add_text("0 FPS")
            # add sliders/text boxes for exposure time and fps
            dpg.bind_font(default_font_small)
            dpg.set_global_font_scale(0.5)

    gui_x_offset = 0
    gui_y_offset = 0
    gui_x_max = 0
    gui_y_max = 0
    row_pos = 0
    for _id, _cam in cameras.items():
        cur_key = f"Camera {_id}"
        dpg.set_item_pos(cur_key, (gui_x_offset, gui_y_offset))

        width = _cam._width // display_downsample + 25
        height = _cam._height // display_downsample + 100

        gui_x_max = int(np.maximum(gui_x_offset + width, gui_x_max))
        gui_y_max = int(np.maximum(gui_y_offset + height, gui_y_max))

        row_pos += 1
        if row_pos == gui_ncols:
            row_pos = 0
            gui_x_offset = 0
            gui_y_offset += height
        else:
            gui_x_offset += width

    [_cam.start_acquisition() for _cam in cameras.values()]



    for _cam in cameras.values():
        _cam.count = 0

    dpg.create_viewport(title="Camera preview", width=gui_x_max, height=gui_y_max)

    # dpg.set_viewport_vsync(False)
    # dpg.show_metrics()
    dpg.setup_dearpygui()
    dpg.show_viewport()


    start_time = -np.inf
    prior_fps = np.nan
    cur_duration = 0


    if record:
        import _thread
        user_input_ts_filename = os.path.join(depth_files_path, f"use_input_ts.txt")
        user_input_ts_file = open(user_input_ts_filename, 'w')
        def get_timestamp(file):
            while True:
                a = input('Enter to log timestamp')
                # get current timestamp
                ts = time.time_ns()
                file.write(f"{ts} \n")
                file.flush() 

        _thread.start_new_thread(get_timestamp, (user_input_ts_file,))

    try:
        while dpg.is_dearpygui_running():
            dat = {}
            for _id, _cam in cameras.items():
                new_frame = None
                new_ts = None

                # do we need a separate thread for this, then grab whatever frame is latest???
                while True:
                    _dat = _cam.try_pop_frame()
                    if _dat[0] is None:
                        break
                    else:
                        if ~np.isfinite(start_time):
                            start_time = time.perf_counter()
                        new_frame = _dat[0]
                        new_ts = _dat[1]
                dat[_id] = (new_frame, new_ts)

            cur_duration = (time.perf_counter() - start_time) / 60.0
            for _id, _dat in dat.items():
                if _dat[0] is not None:
                    disp_min = dpg.get_value(f"texture_{_id}_min")
                    disp_max = dpg.get_value(f"texture_{_id}_max")
                    height, width = _dat[0].shape
                    disp_img = cv2.resize(
                        _dat[0],
                        (width // display_downsample, height // display_downsample),
                    )
                    plt_val = intensity_to_rgba(
                        disp_img,
                        minval=disp_min,
                        maxval=disp_max,
                        colormap=display_colormap,
                    ).astype("float32")
                    cv2.putText(
                        plt_val,
                        str(cameras[_id].frame_count),
                        txt_pos,
                        font,
                        1,
                        (1, 1, 1, 1),
                    )
                    dpg.set_value(f"texture_{cameras[_id].id}", plt_val)
                    cameras[_id].count += 1
                    miss_frames = float(cameras[_id].missed_frames)
                    total_frames = float(cameras[_id].total_frames)
                    cur_fps = cameras[_id].fps
                    # if np.isnan(prior_fps):
                    #     smooth_fps = cur_fps
                    # else:
                    #     smooth_fps = .01 * cur_fps + .99 * prior_fps
                    # prior_fps = smooth_fps

                    try:
                        percent_missed = (miss_frames / total_frames) * 100
                    except:
                        percent_missed = 0

                    dpg.set_value(
                        miss_status[_id],
                        f"{miss_frames} missed / {total_frames} total ({percent_missed:.1f}% missed)",
                    )
                    if cur_fps is not None:
                        dpg.set_value(
                            fps_status[_id],
                            f"{cur_fps:.0f} FPS",
                        )
                    if "storage" in use_queues.keys():
                        for k, v in use_queues["storage"].items():
                            logging.debug(v.qsize())

            if (
                np.isfinite(cur_duration)
                and (duration > 0)
                and (cur_duration > duration)
            ):
                logging.info(f"Exceeded {duration} minutes, exiting...")
                break

            # time.sleep(0.005)
            dpg.render_dearpygui_frame()
    finally:
        [_cam.stop_acquisition() for _cam in cameras.values()]
        if record:
            # for every camera ID wait until the queue has been written out
            print("Issuing stop signal...")
            for k, v in use_queues["storage"].items():
                v.put(None)  # stop signal
                time.sleep(0.1)
                if v.qsize() is not None:
                    while v.qsize() > 0:
                        time.sleep(0.1)
            for _recorder in recorders:
                _recorder.is_running = 0
                time.sleep(1)
        # ensure all files are flushed and closed...
        [_recorder.close_writer() for _recorder in recorders]
        dpg.destroy_context()
        process.wait()


if __name__ == "__main__":
    cli()
