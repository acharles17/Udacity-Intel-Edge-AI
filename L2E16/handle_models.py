import cv2
import numpy as np


def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # TODO 1: Extract only the second blob output (keypoint heatmaps)
    
    print("Length Output is: ", len(output))
    print(output.keys())
    out = output['Mconv7_stage2_L2']
    #out = np.copy((output[1]))
    print("Shape second blob is: ", out.shape)
    print("Input_shape is: ", input_shape)
    #print(out)
    
    
    # TODO 2: Resize the heatmap back to the size of the input
    
    # create array to handle output map
    out_heatmap = np.zeros([out.shape[1], input_shape[0], input_shape[1]])
    for i in range(len(out[0])):
        out_heatmap[i] = cv2.resize(out[0][i], input_shape[0:2][::-1])    
    
    
    #result = out.transpose((1, 19, 32, 57))
    

    #final = cv2.resize(out, (1,19), fx=8, fy=9) # (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)

    #final = cv2.resize(out, (0,0), (input_shape[1], input_shape[0])) #, interpolation=cv2.INTER_AREA
    #final = np.resize(out, input_shape)
    
    #final = out.reshape(input_shape) #[1], input_shape[0])    # [1, 19, 32, 57]
    
    return out_heatmap

#python app.py -i "images/sitting-on-car.jpg" -t "POSE" -m "/home/workspace/models/human-pose-estimation-0001.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
    '''
    # TODO 1: Extract only the first blob output (text/no text classification)
    print("Length Output is: ", len(output))
    print(output.keys())
    out = output['model/segm_logits/add']    
    print("Shape of first blob is: ", out.shape)
    print("Input_shape is: ", input_shape)    
    
    # TODO 2: Resize this output back to the size of the input
    out_text = np.zeros([out.shape[1], input_shape[0], input_shape[1]])
    for i in range(len(out[0])):
        out_text[i] = cv2.resize(out[0][i], input_shape[0:2][::-1])
    return out_text

#python app.py -i "images/sign.jpg" -t "TEXT" -m "/home/workspace/models/text-detection-0004.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def handle_car(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    print("Length Output is: ", len(output))
    print(output.keys())    
    print("Input_shape is: ", input_shape)   
    
    # TODO 1: Get the argmax of the "color" output
    color = output['color']    
    print("Shape of color blob is: ", color.shape)
    
    color = color.flatten()
    
    color_max = np.argmax(color[1])
    
    #col = CAR_COLORS[np.argmax(output[0])]
    # TODO 2: Get the argmax of the "type" output
    type = output['type']    

    print("Shape of type blob is: ", type.shape)
    type = type.flatten()
    type_max = np.argmax(type[1])
    
    #typ = CAR_TYPES[np.argmax(output[1])]
    
    return color_max, type_max

#python app.py -i "images/blue-car.jpg" -t "CAR_META" -m "/home/workspace/models/vehicle-attributes-recognition-barrier-0039.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    elif model_type == "TEXT":
        return handle_text
    elif model_type == "CAR_META":
        return handle_car
    else:
        return None


'''
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
'''
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