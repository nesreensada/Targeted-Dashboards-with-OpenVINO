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
	pass

def pose_detection():
	"""
    Method to extract the head pose for the detected face.

    Args:	
		
    Returns: 
    	
    """
	pass

def age_detection():
	"""
    Method to extract the age of the the detected face.
    Args:

    Returns: 
    	Age of the given face coordinates
    """
	pass

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