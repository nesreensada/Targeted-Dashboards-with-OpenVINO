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

def face_detection(res, conf_threshold, inital_wh):
	"""
    Method to parse Face detection output.
    Args:

    Returns: 
    	list of detected faces and the coordinates for these
    """
    faces = []
    for detected_box in res:
	    confidence = float(detected_box[2])
	    if confidence > conf_threshold:
	        x_min = int(abs(detected_box[3]) * image.shape[1])
	        y_min = int(abs(detected_box[4]) * image.shape[0])
	        x_max = int(detected_box[5] * image.shape[1])
	        y_max = int(detected_box[6] * image.shape[0])
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
    if ((yaw > -22.5) & (yaw < 22.5) & (pitch > -22.5) &
                            (pitch < 22.5)):
    	return True
    else: 
    	return False

def age_detection(age, gender):
	"""
    Method to extract the age of the the detected face.
    Args:

    Returns: 
    	Age of the given face coordinates
    """
	GENDER_TYPE = ["male", "female"]
    age = int(human_age[0] * 100)
    gender_class = GENDER_TYPE[np.argmax(human_gender)]
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

def create_output_image(model_type, image, output, conf_threshold):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    '''
    pass