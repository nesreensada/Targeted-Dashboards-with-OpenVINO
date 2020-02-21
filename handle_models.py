import cv2
import numpy as np



def preprocessing(input_image, height, width):
	'''
	Method to preprocess images, given an input image, height and width:
		- Resize to width and height
		- Transpose the final "channel" dimension to be first
		- Reshape the image to add a "batch" of 1 at the start 
	Args:
		input_image
		height
		width
	Returns:
		preprocessed images
	'''
	image = np.copy(input_image)
	image = cv2.resize(image, (width, height))
	image = image.transpose((2,0,1))
	image = image.reshape(1, 3, height, width)

	return image

def face_detection(res, conf_threshold, initial_wh):
	"""
	Method to parse Face detection output.
	Args:

	Returns: 
		list of detected faces and the coordinates for these
	"""
	faces = []
	for detected_box in res[0][0]:
		confidence = float(detected_box[2])
		if confidence > conf_threshold:
			x_min = int(abs(detected_box[3]) * initial_wh[0])
			y_min = int(abs(detected_box[4]) * initial_wh[1])
			x_max = int(detected_box[5] * initial_wh[0])
			y_max = int(detected_box[6] * initial_wh[1])
			faces.append([x_min, y_min, x_max, y_max])
	return faces

def pose_detection(yaw, pitch):
	"""
	#TODO: this condition needs to be checked 
	Method to extract the onlookers from the 
	yaw and pitch parameters detected from the head pose
	from the model 
	We assume here that if the yaw and pitch needs to be less than 22.5
	or more than -22.5  
	Args:	
		yaw: Yaw is the rotation around the Y-axis.   
		pitch: Pitch around the X-axis
	Returns: 
	"""
	if ((yaw > -30) & (yaw < 30) & (pitch > -30) &
		(pitch < 30)):
		return True
	else: 
		return False

def age_detection(age, gender):
	"""
	Method to extract the age of the the detected face.
	Args:
		age
		gender
	Returns: 
		Age of the given face coordinates
	"""
	GENDER_TYPE = ["male", "female"]
	age = int(age[0] * 100)
	gender_class = GENDER_TYPE[np.argmax(gender)]
	return age, gender_class

def get_mask(processed_output):
	'''
	Given an input image size and processed output for a semantic mask,
	returns a masks able to be combined with the original image.
	'''
	# Create an empty array for other color channels of mask
	empty = np.zeros(processed_output.shape)
	# Stack to make a Green mask where text detected
	mask = np.dstack((empty, processed_output, empty))
	return mask

def create_output_image(image, image_properties):
	'''
	Method that use the detected image properties to add the image specs
	thus it creates an output image showing the result of inference.
	Args:
		image: input image
		image_properties: coordinates of faces, looking, age, gender, head pose angles
	Returns:
		output image showing the result of the inference	
	'''
	for face_id,face_prop in image_properties.items():
		# put the bounary boxs
		x_min, y_min, x_max, y_max = face_prop['coordinates']
		age = face_prop['age']
		gender = face_prop['gender']
		looking = face_prop['looking']
		angle_p = face_prop['pose']['pitch']
		angle_yaw = face_prop['pose']['yaw']
		image = cv2.rectangle(
					image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
		is_looking_text = 'looking' if looking else 'not looking'
		# this size need to be smaller or so depending on the up coordinates
		cv2.putText(image,"Age: {}, Gener: {}, is {}".format(age, gender, is_looking_text),
							(50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
							2 * scaler, (255, 255, 255), 3 * scaler)

	cv2.imwrite("outputs/{}-output.png".format('detected'), image)
