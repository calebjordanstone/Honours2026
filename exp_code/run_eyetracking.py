import tobii_research as tr
from eye_tracking_functions import *
from tobii_research_addons import ScreenBasedCalibrationValidation, Point2
import pandas as pd
import os
import time
import sys
from psychopy import gui, visual, event

# check time in Unix epoch time
time.time()

## ========================================================
## Set paths
## ========================================================
EXPERIMENT = 'valDecode2026'
ID = int(input('\n\n ENTER SUBJECT NUMBER: '))
SUBID = 'sub-' + f'{ID}'.zfill(2)
DATAPATH = 'C:/Experiments/valDecode2026/data/'
os.makedirs(DATAPATH + SUBID + '/eye/')
## ========================================================
## Connect to eyetracker
## ========================================================
found_eyetrackers = tr.find_all_eyetrackers()
if len(found_eyetrackers) == 0:
    print("\n\nNO EYETRACKER FOUND!\n\n")
    sys.exit()
else:
    my_et = found_eyetrackers[0]
my_et.get_gaze_output_frequency() # 600
## ========================================================
## Configre eye tracker
## ========================================================
# ## set the display area
display_area = my_et.get_display_area()
# my_et.set_display_area()

# set using user coordinate system
#z = points toward user perpendicular to the front surfice of the eye tracker
#y = points vertically towards the user's up
#x = points hoizontally towards the user's right
topLeft = (-302, 433, -95)
bottomLeft = (-302, 79, -95)
topRight = (302, 433, -95)
new_display_area_dict = dict()
new_display_area_dict['top_left'] = topLeft
new_display_area_dict['top_right'] = topRight
new_display_area_dict['bottom_left'] = bottomLeft
new_display_area = tr.DisplayArea(new_display_area_dict)
my_et.set_display_area(new_display_area)

## ========================================================
## Initial calibration
## ========================================================
accepted_inputs = ['0', '1', '2']
while True:
    ENTER_CALIBRATION = input('\n\n PRESS 1 TO ENTER CALIBRATION. \n PRESS 2 TO SKIP. \n PRESS 0 TO EXIT. \n\n')
    if ENTER_CALIBRATION in accepted_inputs:
        ENTER_CALIBRATION = int(ENTER_CALIBRATION)
        break
    else: 
        print('Input not valid! Try again')
enter_calibration(ENTER_CALIBRATION, my_et)
## ========================================================
## Initial validation
## ========================================================
while True:
    ENTER_VALIDATION = input('\n\n PRESS 1 TO ENTER VALIDATION. \n PRESS 2 TO SKIP. \n PRESS 0 TO EXIT. \n\n')
    if ENTER_VALIDATION in accepted_inputs:
        ENTER_VALIDATION = int(ENTER_VALIDATION)
        break
    else: 
        print('Input not valid! Try again')
enter_validation(ENTER_VALIDATION, my_et)
## ========================================================
## Get gaze data
## ========================================================
print('STARTING RECORDING')
# create data structure to store data
gaze_data_buffer = []
external_signal_buffer = []
winsize = [1920, 1080]
trig = 0
block = 1
# define callback function (runs each time a new data point is collected, allows us to extract the data we want from the gaze_data object)
def gaze_data_callback(gaze_data):

    global gaze_data_buffer
    global trig
    global block 
    global winsize # adjust for differnet coordinates - check this!!

    # Extract the data we are interested in
    t  = gaze_data.system_time_stamp / 1000
    lx = gaze_data.left_eye.gaze_point.position_on_display_area[0] * winsize[0]
    ly = gaze_data.left_eye.gaze_point.position_on_display_area[1] * winsize[1]
    lp = gaze_data.left_eye.pupil.diameter
    lv = gaze_data.left_eye.gaze_point.validity
    rx = gaze_data.right_eye.gaze_point.position_on_display_area[0] * winsize[0]
    ry = gaze_data.right_eye.gaze_point.position_on_display_area[1] * winsize[1]
    rp = gaze_data.right_eye.pupil.diameter
    rv = gaze_data.right_eye.gaze_point.validity
    trigger = trig
    block_no = block

    # Save data
    gaze_data_buffer.append((block_no, t, lx, ly, lp, lv, rx, ry, rp, rv, trigger))

    # Reset so that trigger value appears only once
    trigger = '' 

def external_signal_callback(external_signal_data):

    # global external_signal_buffer
    global trig

    # Extract the trigger value
    trigger = external_signal_data.value

    # Update global trigger variable
    trig = trigger
    print(trig)

# function to save data file
def write_buffer_to_file(gaze_data_buffer, output_path): # external_signal_buffer

    # Make a copy of the buffer and clear it
    gaze_data_buffer_copy = gaze_data_buffer[:]
    gaze_data_buffer.clear()
    
    # Define column names for the dataframe
    gaze_data_columns = ['block', 'time_gd', 'L_X', 'L_Y', 'L_P', 'L_V', 
                         'R_X', 'R_Y', 'R_P', 'R_V', 'event'] 

    # Convert buffer to DataFrame
    gaze_data_out = pd.DataFrame(gaze_data_buffer_copy, columns=gaze_data_columns)

    # Check if the file exists
    file_exists = os.path.isfile(output_path)
    
    # Write the DataFrame to a csv file
    gaze_data_out.to_csv(output_path, mode='a', index =False, header = not file_exists)

# start recording
my_et.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
my_et.subscribe_to(tr.EYETRACKER_EXTERNAL_SIGNAL, external_signal_callback)

# get input to save data
while True:

    # save data automatically at end of block
    if trig == 254:
        FILENAME = f'{SUBID}_task-{EXPERIMENT}_eye_block-{block}.csv'
        FILEPATH = DATAPATH + SUBID + '/eye/' + FILENAME
        time.sleep(1)
        write_buffer_to_file(gaze_data_buffer, FILEPATH) # external signal buffer
        print('Saving data')
        block += 1 
    # exit recording
    if trig == 255:
        print('Exiting')
        break

# unsubscribe from callbacks
my_et.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
my_et.unsubscribe_from(tr.EYETRACKER_EXTERNAL_SIGNAL, external_signal_callback)

# display reminder to redo calibration 
# dlg = gui.Dlg(title="")
# dlg.addText("Redo calibration and validation")
# dlg.show()

win = visual.Window(fullscr=1)
txt = visual.TextStim(win = win, 
                      text= "Redo calibration and validation. \n\n Press SPACE to continue")
while True:

    txt.draw()
    win.flip()
    pressed = event.getKeys(keyList='space')
    if pressed:
        break
win.close()
## ========================================================
## Second calibration
## ========================================================
while True:
    ENTER_CALIBRATION = input('\n\n PRESS 1 TO ENTER CALIBRATION. \n PRESS 2 TO SKIP. \n PRESS 0 TO EXIT. \n\n')
    if ENTER_CALIBRATION in accepted_inputs:
        ENTER_CALIBRATION = int(ENTER_CALIBRATION)
        break
    else: 
        print('Input not valid! Try again')
enter_calibration(ENTER_CALIBRATION, my_et)

## ========================================================
## Second validation
## ========================================================
while True:
    ENTER_VALIDATION = input('\n\n PRESS 1 TO ENTER VALIDATION. \n PRESS 2 TO SKIP. \n PRESS 0 TO EXIT. \n\n')
    if ENTER_VALIDATION in accepted_inputs:
        ENTER_VALIDATION = int(ENTER_VALIDATION)
        break
    else: 
        print('Input not valid! Try again')
enter_validation(ENTER_VALIDATION, my_et)

# restart recording 
print('STARTING RECORDING')
trig = 0
my_et.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
my_et.subscribe_to(tr.EYETRACKER_EXTERNAL_SIGNAL, external_signal_callback)

# get input to save data
while True:

    # save data automatically at end of block
    if trig == 254:
        FILENAME = f'{SUBID}_task-{EXPERIMENT}_eye_block-{block}.csv'
        FILEPATH = DATAPATH + SUBID + '/eye/' + FILENAME
        time.sleep(1)
        write_buffer_to_file(gaze_data_buffer, FILEPATH) # external signal buffer
        print('Saving data')
        block += 1 
    # exit recording
    if trig == 255:
        print('Exiting')
        break

# stop recording
my_et.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
my_et.unsubscribe_from(tr.EYETRACKER_EXTERNAL_SIGNAL, external_signal_callback)
print('ENDING RECORDING')






















































