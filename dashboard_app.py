import argparse
import cv2
import numpy as np
import time
import json 
import os
import sys
from inference import Network

import logging as log

# Global variables 
CONFIG_FILE = '../resources/config.json'
accepted_devices = ['CPU', 'GPU', 'MYRIAD', 'HETERO:FPGA,CPU', 'HDDL']
is_async_mode = True
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


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
    required.add_argument("-fm", help=fm_desc, required=True)
    required.add_argument("-pm", help=pm_desc, required=True)
    required.add_argument("-ag_m", help=agm_desc, required=True)

    optional.add_argument("-l", "--cpu_extension", type=str, default=CPU_EXTENSION,
                        help="extension for the CPU device")
    #optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-ct", help=conf_desc, default=0.5, type=float)
    parser.add_argument("-f", "--flag", help=req_desc, default="async", type=str)

    args = parser.parse_args()

    return args

def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image

def face_detection():
	"""
    Method to parse Face detection output.
    Args:
    Returns: 
    """
	pass

def main():
	"""
    Method to load the network and parse the output.
    Args: 
    	None
    Returns:
    	None
    """
	pass



if __name__ == '__main__':
    main()
    sys.exit()