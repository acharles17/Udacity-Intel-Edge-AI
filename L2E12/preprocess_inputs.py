import cv2
import numpy as np

def preprocessing(input_image, height, width):
    
    #print("original: ", input_image)

    # to rescale the image
    image = cv2.resize(input_image, (width, height))
    
    # to transpose and put width first
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, 3, height, width)
    
    #print("rescaled: ", image)
    return image


def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)
    
    #print(preprocessed_image)
    # TODO: Preprocess the image for the pose estimation model
    #image = cv2.imread(preprocessed_image, cv2.IMREAD_COLOR)
    #image2 = cv2.resize(image)
    preprocessed_image = preprocessing(preprocessed_image, 256, 456)
    #cv2.imshow(preprocessed_image)
    return preprocessed_image

#img_path = r'/home/workspace/images/'

#pose_estimation(img_path + 'sitting-on-car.jpg')


def text_detection(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the text detection model
    preprocessed_image = preprocessing(preprocessed_image, 768, 1280)

    return preprocessed_image


def car_meta(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the car metadata model
    preprocessed_image = preprocessing(preprocessed_image, 72, 72)

    return preprocessed_image
