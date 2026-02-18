import numpy as np
# from pypixxlib import _libdpx as dp

# Creating new stimulus positions to display using PROPIXX QUAD4x mode
def reformat_for_propixx(win, position):

        # define new stimulus positions
        x = win.size[0]/4
        y = win.size[1]/4
        offsets = [[-x,y], [x,y], [-x,-y], [x, -y]]
        
        # find new positions
        rescaled_positions = np.array(position)/2
        new_positions = [] # create empty list to store new positions
        for quadrant in range(0,4): 
            new_stim_pos = [rescaled_positions[0] + offsets[quadrant][0], # x-axis
                            rescaled_positions[1] + offsets[quadrant][1]] # y-axis
            new_positions.append(new_stim_pos)
        
        return new_positions

# Send triggers using DATApixx    
# def send_trigger(trig_value, update=False):

#     # set all pins to 'on'
#     bit_mask = 0xFFFFFF    

#     # set trigger value
#     dp.DPxSetDoutValue(trig_value, bit_mask)

#     # write or update register cache
#     if not update:
#         dp.DPxWriteRegCache() 
#     else:
#         dp.DPxUpdateRegCache()

# Find desired size of stimuli in pixels
def dva_to_pix(dva, distance, screen_res, screen_width):

    # convert from dva to stimulus size in cms 
    # https://www.khanacademy.org/math/geometry-home/right-triangles-topic/trig-solve-for-a-side-geo/a/unknown-side-in-right-triangle-w-trig
    # https://www.sr-research.com/eye-tracking-blog/background/visual-angle/
    stim_size = np.tan(np.deg2rad(dva))*distance
    # find screen resolution of propixx
    # screen_res_x = screen_res[0]/2
    # find size of each pixel
    pixel_size = screen_width / screen_res[0]
    # find how many pixels are needed for object of given size in cm
    pixels = int(stim_size/pixel_size)

    return np.abs(pixels)
