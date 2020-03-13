import argparse
import cv2
from inference import Network
import numpy as np

INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ci_desc = "The confidence thresholds used to draw bounding boxes"
    ###       2) The user choosing the color of the bounding boxes
    col_desc = "The desired color of the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    #optional.add_argument("-ci", help=ci_desc, default='CPU')
    #optional.add_argument("-col", help=col_desc, default='CPU')
    args = parser.parse_args()

    return args


def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    return frame


def infer_on_video(args):
    ### TODO: Initialize the Inference Engine
    plugin = Network()
    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    print((width,height))
    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        #cv2.add(frame
        #preprocessed_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        #def preprocessing(input_image, height, width):
        '''
        Given an input image, height and width:
        - Resize to width and height
        - Transpose the final "channel" dimension to be first
        - Reshape the image to add a "batch" of 1 at the start 
        '''
        '''
        image = np.copy(frame)
        print(image.shape)
        '''
        
        image = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        image = image.transpose((2,0,1))
        
        preprocessed_image = image.reshape(1, *image.shape)
        #print((width,height))
        #return image 
        #cv2.resize(frame, (width, height))    
        #cv2.imshow(preprocessed_image, cap)
        
        
        
        ### TODO: Perform inference on the frame
        print(plugin.network.inputs)
        plugin.async_inference(preprocessed_image)
        ### TODO: Get the output of inference
        if plugin.wait() == 0:
            res = plugin.extract_output()

            ### TODO: Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, res, args, width, height)

            # Write out the frame
            out.write(frame)
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
