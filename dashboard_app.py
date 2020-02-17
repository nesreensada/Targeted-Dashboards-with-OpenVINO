import argparse
import cv2
import numpy as np
import time
import json
import os
import sys
from inference import Network
import logging as log

import handle_models

# Global variables
CONFIG_FILE = '../resources/config.json'
accepted_devices = ['CPU', 'GPU', 'MYRIAD', 'HETERO:FPGA,CPU', 'HDDL']
is_async_mode = True
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
image_flag = False


def get_args():
    """
    Method to extract the arguments from the command line
    """
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    fm_desc = "The location of the face detection model XML file"
    pm_desc = "The location of the pretrained head pose model XML file"
    agm_desc = "The location of the pretrained age gender detection model XML file"

    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    conf_desc = "The thershold for expected image thershold"
    req_desc = "The request hanlding type"
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-fm", "--facemodel", help=fm_desc, required=True)
    required.add_argument("-pm", "--posemodel", help=pm_desc, required=True)
    required.add_argument("-ag_m", "--agemodel", help=agm_desc, required=True)

    optional.add_argument("-l", "--cpu_extension", type=str, default=CPU_EXTENSION,
                        help="extension for the CPU device")
    # optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", "--device", help=d_desc, default='CPU')
    optional.add_argument("-ct", "--confidence",
                          help=conf_desc, default=0.5, type=float)
    parser.add_argument("-f", "--flag", help=req_desc,
                        default="async", type=str)

    args = parser.parse_args()
    if args.device not in accepted_devices:
    	print("Unsupported Device: " + args.device,
    	      "the accepted devices are as follows" + accepted_devices)
    	print("Unsupported Device: {}, accepted_devices: {}".format(
    	    args.device, accepted_devices))
   		sys.exit(1)
   	if args.flag == "sync":
        is_async_mode = False
    else:
        is_async_mode = True
    return args


def main():
	"""
    Method to load the network and parse the output.
    Args:
    	None
    Returns:
    	None
    """
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = args_parser().parse_args()
    logger = log.getLogger()
    assert os.path.isfile(CONFIG_FILE), "{} file doesn't exist".format(CONFIG_FILE)
    config = json.loads(open(CONFIG_FILE).read())
    
    # Decide the input stream based on the config video item
    item = config['input']['video']

    # TODO: add image as an option for the input stream 
    if item.isdigit():
    	# CAM 
    	input_stream = int(item)
    else:
    	# video or images
    	if item.endswith('.jpg') or item.endswith('.bmp') or item.endswith('.png'):
    		# TODO: handle images
    		image_flag =  True
    	input_stream = item

    # Create a Network for using the Inference EngineNetwork for each model 
    # 1. Face detection
    infer_network_fd = Network()
    # Load the model in the network, and obtain its input shape
    fd_plugin, (n_fd, c_fd, h_fd, w_fd) = infer_network_fd.load_model(args.facemodel, args.device, args.cpu_extension)

    # 2. head pose 
    infer_network_pose = Network()
    # Load the model in the network, and obtain its input shape
    n_p, c_p, h_p, w_p = infer_network_pose.load_model(args.posemodel, args.device, args.cpu_extension, fd_plugin)[1]

    # 3. age (TODO: check if the plugin is added like this or no)
    infer_network_age = Network()
    # Load the model in the network, and obtain its input shape
    n_a, c_a, h_a, w_a = infer_network_age.load_model(args.agemodel, args.device, args.cpu_extension, fd_plugin)[1]



    # read the input stream 
    if input_stream:
        cap.open(input_stream)
        # Adjust DELAY to match the number of FPS of the video file
        DELAY = 1000 / cap.get(cv2.CAP_PROP_FPS)

    # TODO: maybe remove the next 2 lines 
    # Init inference request IDs
    cur_request_id = 0
    next_request_id = 1

    # TODO: maybe remove sync
    if is_async_mode:
        print("Application running in async mode...")
    else:
        print("Application running in sync mode...")

    # performance bottlenecks
    det_time_fd = 0
    ret, frame = cap.read()
    while ret:
    	# number of people looking
    	looking = 0
    	ret, frame = cap.read()
        if not ret:
            KEEP_RUNNING = False
            break
        if frame is None:
            KEEP_RUNNING = False
            log.error("ERROR! blank FRAME grabbed")
            break

        initial_wh = [cap.get(3), cap.get(4)]
        # process the f_d model
        # preprocessing(input_image, height, width)
        in_frame_fd = handle_models.preprocessing(frame, h_fd, w_fd)

        key_pressed = cv2.waitKey(int(DELAY))

        # Start asynchronous inference for specified request
        inf_start_fd = time.time()

        if is_async_mode:
            # Async enabled and only one video capture
            infer_network_fd.exec_net(next_request_id, in_frame_fd)
        else:
            # Async disabled
            infer_network_fd.exec_net(cur_request_id, in_frame_fd)

        # Wait for the result
        if infer_network.wait(cur_request_id) == 0:
            det_time_fd = time.time() - inf_start_fd
        # Results of the output layer of the network
            res = infer_network.get_output(cur_request_id)
            # Parse face detection output
            # TODO: the below is to be implemented
            faces = handle_models.face_detection(res, args, initial_wh)
            # here we need to add handling the poses 
            # then extracting the age ?? for items that are looking in the correct direction?
	pass



if __name__ == '__main__':
    main()
    sys.exit()
