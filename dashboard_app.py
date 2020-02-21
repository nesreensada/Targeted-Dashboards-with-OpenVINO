import argparse
import cv2
import numpy as np
import time
import json
import os
import sys
from inference import Network
import logging as log
import paho.mqtt.client as mqtt
import socket

import handle_models

# Global variables
CONFIG_FILE = 'resources/config.json'

accepted_devices = ['CPU', 'GPU', 'MYRIAD', 'HETERO:FPGA,CPU', 'HDDL']
is_async_mode = True
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
image_flag = False

# MQTT server environment variables

MQTT_HOST = "localhost"
topic = "test/message"

DEFAULT_DATA = {"dashboard": "kids"}

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
	#optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
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
	args = get_args()
	if args.flag == "sync":
		is_async_mode = False
	else:
		is_async_mode = True
	logger = log.getLogger()
	assert os.path.isfile(CONFIG_FILE), "{} file doesn't exist".format(CONFIG_FILE)
	config = json.loads(open(CONFIG_FILE).read())
	
	# Decide the input stream based on the config video item
	item = config['input']['video']
	image_flag = False

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

	# TODO: uncomment this part for the MQTT connection
	# Connect to the MQTT server
	client = mqtt.Client()
	client.connect(MQTT_HOST)

	cap = cv2.VideoCapture(input_stream)
	# read the input stream 
	if input_stream:
		cap.open(input_stream)
		# Adjust DELAY to match the number of FPS of the video file
		#DELAY = 1000 / cap.get(cv2.CAP_PROP_FPS)

	if not cap.isOpened():
		logger.error("ERROR! Unable to open video source")

	# Init inference request IDs
	cur_request_id = 0
	next_request_id = 1

	# Create a Network for using the Inference EngineNetwork for each model 
	# 1. Face detection
	infer_network_fd = Network()
	# Load the model in the network, and obtain its input shape
	fd_plugin, (n_fd, c_fd, h_fd, w_fd) = infer_network_fd.load_model(args.facemodel, args.device, 0, args.cpu_extension)

	# 2. head pose 
	infer_network_pose = Network()
	# Load the model in the network, and obtain its input shape
	n_p, c_p, h_p, w_p = infer_network_pose.load_model(args.posemodel, args.device, 0, args.cpu_extension, fd_plugin)[1]

	# 3. age (TODO: check if the plugin is added like this or no)
	infer_network_age = Network()
	# Load the model in the network, and obtain its input shape
	n_a, c_a, h_a, w_a = infer_network_age.load_model(args.agemodel, args.device, 0, args.cpu_extension, fd_plugin)[1]

	# TODO: maybe remove sync
	if is_async_mode:
		print("Application running in async mode...")
		logger.info("Application running in async mode...")
	else:
		print("Application running in sync mode...")
		logger.info("Application running in sync mode...")
	# performance bottlenecks
	det_time_fd = 0
	ret, frame = cap.read()
	while ret:
		# number of people looking
		looking = 0
		ret, frame = cap.read()

		if not ret: 
			log.error("ERROR! blank FRAME grabbed")
			break
		if frame is None:
			log.error("ERROR! blank FRAME grabbed")
			break

		initial_wh = [cap.get(3), cap.get(4)]
		# process the f_d model
		# preprocessing(input_image, height, width)
		in_frame_fd = handle_models.preprocessing(frame, h_fd, w_fd)

		key_pressed = cv2.waitKey(60)

		# Start asynchronous inference for specified request
		inf_start_fd = time.time()
		infer_network_fd.exec_net(0, in_frame_fd)

		people_dict = {}
		# Wait for the result
		if infer_network_fd.wait(cur_request_id) == 0:
			det_time_fd = time.time() - inf_start_fd
		# Results of the output layer of the network
			res = infer_network_fd.get_output(cur_request_id)
			# Parse face detection output
			faces = handle_models.face_detection(res, args.confidence, initial_wh)
			logger.info('how many faces {}'.format(len(faces)))
			# then extracting the age ?? for items that are looking in the correct direction?
			# if we have one person older than 25 and looking then send to trigger older dashboard
			if len(faces) > 0:
				# look for people poses and the age of people also
				for face_id, face_loc in enumerate(faces):

					xmin, ymin, xmax, ymax = face_loc
					face_frame = frame[ymin:ymax, xmin:xmax]

					# preprocessing for headpose and age models
					in_frame_hp = handle_models.preprocessing(face_frame, w_p, h_p)
					in_frame_age = handle_models.preprocessing(face_frame, w_a, h_a)
					

					inf_start_hp = time.time()
					infer_network_pose.exec_net(cur_request_id, in_frame_hp)
					infer_network_pose.wait(cur_request_id)
					det_time_hp = time.time() - inf_start_hp

					# Parse head pose detection results
					# pitch angle: Pitch around the X-axis
					angle_p_fc = infer_network_pose.get_output(0, "angle_p_fc").flatten()[0]
					# yaw pose: Yaw is the rotation around the Y-axis.   
					angle_y_fc = infer_network_pose.get_output(0, "angle_y_fc").flatten()[0]

					# this needs to be moved to the decision stage and preporcessing 
					looking_flag = handle_models.pose_detection(yaw = angle_y_fc, pitch=angle_p_fc)

					# age detection 
					inf_start_a = time.time()
					infer_network_age.exec_net(cur_request_id, in_frame_age)
					infer_network_age.wait(cur_request_id)
					det_time_a = time.time() - inf_start_a

					# check if i need to flatten or no
					human_age = infer_network_age.get_output(0, "age_conv3").flatten()
					human_gender = infer_network_age.get_output(0, "prob").flatten()
					age, gender = handle_models.age_detection(human_age, human_gender)
					people_dict[face_id] = {'coordinates': face_loc, 'pose':{'yaw':angle_y_fc, 'pitch':angle_p_fc},
					 'age': age, 'gender':gender, 'looking':looking_flag}
				# stats messages
				inf_time_message = "Face Inference time: N\A for async mode" if is_async_mode else \
				"Inference time: {:.3f} ms".format(det_time_fd * 1000)
				logger.info(inf_time_message)
				head_inf_time_message = "Head pose Inference time: N\A for async mode" if is_async_mode else \
					"Inference time: {:.3f} ms".format(det_time_hp * 1000)
				logger.info(head_inf_time_message)
				age_inf_time_message = "Age Gender Inference time: N\A for async mode" if is_async_mode else \
					"Inference time: {:.3f} ms".format(det_time_a * 1000)
				logger.info(head_inf_time_message)

				data = DEFAULT_DATA

				logger.info('detected_people {}'.format(people_dict))
				# send the decision to the dashboard based on the people detected
				for people_id, poeple_prop in people_dict.items():
					if poeple_prop['age'] >= 25 and poeple_prop['looking']:
						data = {"dashboard": "adult"}
						break 
				logger.info("data sent to the client is {}".format(data))
				client.publish(topic, json.dumps(data))

			else:
				print("Default Dashboard since we don't have any on lookers")
				logger.info("Default Dashboard since we don't have any on lookers")
				logger.info("data sent to the client is {}".format(DEFAULT_DATA))
				client.publish(topic, json.dumps(DEFAULT_DATA))
		if key_pressed == 27: #add something to break the code
			logger.info("Attempting to stop background threads")
			break
		if image_flag:
			handle_models.create_output_image(frame, people_dict)


	infer_network_fd.clean()
	infer_network_pose.clean()
	infer_network_age.clean()
	cap.release()
	cv2.destroyAllWindows()
	client.disconnect()


if __name__ == '__main__':
	main()
	sys.exit()
