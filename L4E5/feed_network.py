import argparse
### TODO: Load the necessary libraries
#import IECore
#import IENetwork
from openvino.inference_engine import IECore
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin

ver = IECore()
#ver.descr
#print(ver.__version__)

#print(ver.get_versions("CPU")["CPU"].build_number)

#print("{descr}: {maj}.{min}.{num}".format(descr=ver.descr, maj=ver.major, min=ver.minor, num=ver.build_number))

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()

    return args


def load_to_IE(model_xml):
    ### TODO: Load the Inference Engine API
    ie = IECore() 
    
    print(model_xml)
    
    ### TODO: Load IR files into their related class
    net = IENetwork(model=model_xml + ".xml", weights=model_xml + ".bin")
    #print(net.inputs)
    
    '''
    print(net.inputs['data'].precision)
    print(net.inputs['data'].layout)

    print(net.inputs['data'].shape)
    '''
    
    ### TODO: Add a CPU extension, if applicable. It's suggested to check
    ###       your code for unsupported layers for practice before 
    ###       implementing this. Not all of the models may need it.
    
    b1 = net.layers 
    #ie = IECore()
    #ie.add_extension(extension_path="/some_dir/libcpu_extension_avx2.so", device_name="CPU")
    
    ie.add_extension(extension_path=CPU_EXTENSION, device_name="CPU")
    
    b2 = net.layers 
    
    ### TODO: Get the supported layers of the network
    #print((net.layers).keys())
    
    '''
    print("Available layers are: ", (net.layers))

    print("Same layers??? ", b1 == b2)
    '''
    
    ### Get the supported layers of the network
    supported_layers = ie.query_network(network=net, device_name="CPU")
    
        ### Check for any unsupported layers, and let the user
    ### know if anything is missing. Exit the program, if so.
    unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(unsupported_layers) != 0:
        print("Unsupported layers found: {}".format(unsupported_layers))
        print("Check whether extensions are available to add to IECore.")
        exit(1)
        
    ie.load_network(net, "CPU")
    
    """        
    ### TODO: Check for any unsupported layers, and let the user
    ###       know if anything is missing. Exit the program, if so.

    for device in ["CPU" , "GPU", "FPGA", "MYRIAD", "HETERO"]: #
        print("Using %s device." %device)
        
        '''
        '''
        try:
            plugin = IECore(device=device)

            exec_net = plugin.load(network=net, num_requests=2)

                #if exec_net
            message = "IR successfully loaded into Inference Engine."
            print(message)
            #print(exec_net)
            return exec_net
        
        except:
            print("Device missing for loading into Inference Engine.")
            break

        '''
        '''
        #try:
        plugin = IEPlugin(device=device)
        #print("plugin: ", plugin)
        if plugin:

                ### TODO: Load the network into the Inference Engine

            exec_net = plugin.load(network=net, num_requests=2)

                #if exec_net
            message = "IR successfully loaded into Inference Engine."
                
                #return message

        else: #except:
            message = "Device missing for loading into Inference Engine."
            print(message)
            break
        #print(message)         
    #return exec_net'''
    """
    print("IR successfully loaded into Inference Engine.")
    return


'''
python3 feed_network.py -m models/human-pose-estimation-0001
python3 feed_network.py -m models/text-detection-0004
python3 feed_network.py -m models/vehicle-attributes-recognition-barrier-0039
'''


def main():
    args = get_args()
    load_to_IE(args.m)


if __name__ == "__main__":
    main()
