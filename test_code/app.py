import argparse
import cv2
import numpy as np

from handle_models import handle_output, preprocessing
from inference import Network

CAR_COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
CAR_TYPES = ["car", "bus", "truck", "van"]
GENDER_TYPE = ["male", "female"]


def get_args():
    """
    Method to extract the arguments from the command line
    """
    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")

    # create descriptions for the commands
    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m_desc = "The location of the model XML file"
    # TODO: this needs to change according to our models
    t_desc = "The type of model: POSE, TEXT, TEXT_REC, Face, HEAD_POSE or CAR_META"

    conf_threshold = "The thershold for expected image thershold"
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-m", help=m_desc, required=True)
    required.add_argument("-t", help=t_desc, required=True)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    optional.add_argument("-ct", help=conf_threshold, default=0.5, type=float)
    args = parser.parse_args()

    return args


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
    if model_type == "POSE":
        # Remove final part of output not used for heatmaps
        output = output[:-1]
        # Get only pose detections above 0.5 confidence, set to 255
        for c in range(len(output)):
            output[c] = np.where(output[c] > 0.5, 255, 0)
        # Sum along the "class" axis
        output = np.sum(output, axis=0)
        # Get semantic mask
        pose_mask = get_mask(output)
        # Combine with original image
        image = image + pose_mask
        return image
    elif model_type == "TEXT":
        # Get only text detections above 0.5 confidence, set to 255
        output = np.where(output[1] > 0.5, 255, 0)
        # Get semantic mask
        text_mask = get_mask(output)
        # Add the mask to the image
        image = image + text_mask
        return image
    elif model_type == "CAR_META":
        # Get the color and car type from their lists
        color = CAR_COLORS[output[0]]
        car_type = CAR_TYPES[output[1]]
        # Scale the output text by the image shape
        scaler = max(int(image.shape[0] / 1000), 1)
        # Write the text of color and type onto the image
        image = cv2.putText(image,
                            "Color: {}, Type: {}".format(color, car_type),
                            (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                            2 * scaler, (255, 255, 255), 3 * scaler)
        return image
    elif model_type == "AGE_GEN":
        # Get the age and gender from their lists
        age = output[0]
        gender = GENDER_TYPE[output[1]]
        # Scale the output text by the image shape
        scaler = max(int(image.shape[0] / 1000), 1)
        # Write the text of color and type onto the image
        image = cv2.putText(image,
                            "Color: {}, Type: {}".format(age, gender),
                            (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                            2 * scaler, (255, 255, 255), 3 * scaler)
        return image
    elif model_type == "TEXT_REC":
        # I dont know how to handle the ouput
        text = output[0]
        # Scale the output text by the image shape
        scaler = max(int(image.shape[0] / 1000), 1)
        # Write the text of color and type onto the image
        #.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        image = cv2.putText(image,
                            "Text: {}".format(text),
                            (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                            2 * scaler, (255, 255, 255), 3 * scaler)
        return image
    elif model_type == 'FACE':
        for detected_box in output:
            image_id = detected_box[0]
            label = int(detected_box[1])
            confidence = float(detected_box[2])
            if confidence > conf_threshold:
                x_min = int(detected_box[3] * image.shape[1])
                y_min = int(detected_box[4] * image.shape[0])
                x_max = int(detected_box[5] * image.shape[1])
                y_max = int(detected_box[6] * image.shape[0])
                image = cv2.rectangle(
                    image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return image
    elif model_type == 'HEAD_POSE':
        # for detected_box in output:
        #     image_id = detected_box[0]
        #     label = int(detected_box[1])
        #     confidence = float(detected_box[2])
        #     if confidence > conf_threshold:
        #         x_min = int(detected_box[3] * image.shape[1])
        #         y_min = int(detected_box[4] * image.shape[0])
        #         x_max = int(detected_box[5] * image.shape[1])
        #         y_max = int(detected_box[6] * image.shape[0])
        #         image = cv2.rectangle(
        #             image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return image  
    else:
        print("Unknown model type, unable to create output image.")
        return image


def perform_inference(args):
    '''
    Performs inference on an input image, given a model.
    '''
    # Create a Network for using the Inference Engine
    inference_network = Network()
    # Load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)
    conf_threshold = args.ct

    # Read the input image
    image = cv2.imread(args.i)

    # TODO: Preprocess the input image
    preprocessed_image = preprocessing(image, h, w)

    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)

    # Obtain the output of the inference request
    output = inference_network.extract_output()

    # TODO: Handle the output of the network, based on args.t
    # Note: This will require using `handle_output` to get the correct
    # function, and then feeding the output to that function.
    output_func = handle_output(args.t)
    processed_output = output_func(output, image.shape)


    # Create an output image based on network
    output_image = create_output_image(args.t, image, processed_output, conf_threshold)

    # Save down the resulting image
    cv2.imwrite("outputs/{}-output.png".format(args.t), output_image)


def main():
    args = get_args()
    perform_inference(args)


if __name__ == "__main__":
    main()
