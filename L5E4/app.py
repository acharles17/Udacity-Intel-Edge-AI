import argparse
import cv2
import numpy as np

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Handle an input stream")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc)
    args = parser.parse_args()

    return args


def capture_stream(args):
    ### TODO: Handle image, video or webcam
    stream = cv2.VideoCapture(args.i)
    

    ### TODO: Get and open video capture
    
    #videoCapture = cv2.VideoCapture()
    fps = 30
    size = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)
       
    success, frame = stream.read()
    noFramesRemaining = 100 * fps - 1
    while success and noFramesRemaining > 0:
        videoWriter.write(frame)
        success, frame = stream.read()
        noFramesRemaining -= 1
    
        #print("frame is: ", frame)
        
        ### TODO: Re-size the frame to 100x100
        frame = cv2.resize(frame, (100, 100))
        #stream = cv2.resize(2,0,1)
    
    ### TODO: Add Canny Edge Detection to the frame, 
    ###       with min & max values of 100 and 200
    ###       Make sure to use np.dstack after to make a 3-channel image
    edges = cv2.Canny(frame, 100, 200)
    
    img = np.dstack(frame) #, 3
    
    ### TODO: Write out the frame, depending on image or video
    cv2.videoWriter('out.mp4')
    
    ### TODO: Close the stream and any windows at the end of the application
    #cv2.CloseAll()
    cv2.WindowClose()

#'python app.py -i test_video.mp4'
def main():
    args = get_args()
    capture_stream(args)


if __name__ == "__main__":
    main()
