import tobii_research as tr
from tobii_research_addons import ScreenBasedCalibrationValidation, Point2
import pandas as pd
import os
import time
import sys

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
topLeft = (-302, 433, -95) # measured manually
bottomLeft = (-302, 79, -95)
topRight = (302, 433, -95)
new_display_area_dict = dict()
new_display_area_dict['top_left'] = topLeft
new_display_area_dict['top_right'] = topRight
new_display_area_dict['bottom_left'] = bottomLeft
new_display_area = tr.DisplayArea(new_display_area_dict)
my_et.set_display_area(new_display_area)

## ========================================================
## Calibrate
## ========================================================
ENTER_CALIBRATION = int(input('\n\n PRESS 1 TO ENTER CALIBRATION. \n PRESS 2 TO SKIP. \n PRESS 0 TO EXIT. \n\n'))

if ENTER_CALIBRATION == 1:

    calibration_status = [False, False, False, False, False]
    points_to_calibrate = [(0.2, 0.2), # upper left
                           (0.8, 0.2), #upper right
                           (0.5, 0.5), # middle
                           (0.2, 0.8), # lower left
                           (0.8, 0.8)] # lower right
    point_names = ['Top Left', 'Top Right', 'Middle', 'Bottom Left', 'Bottom Right']
        
    calibration = tr.ScreenBasedCalibration(my_et)
    calibration.enter_calibration_mode()

    while True:

        # cycle through points to calibrate on
        for i, point in enumerate(points_to_calibrate):

            print(f'Calibrating position {point_names[i]}')
            while True:
                STATUS = calibration.collect_data(point[0], point[1])
                if STATUS == 'calibration_status_success':
                    calibration_status[i] = True
                NEXT_POINT = int(input(f'\nCALIBRATION POINT STATUS: {STATUS} \n\n PRESS 1 TO CONTINUE. PRESS 0 TO REPEAT. \n\n'))
                if NEXT_POINT:
                    break
                else:
                    calibration.discard_data(point[0], point[1])

        # ask if user wants to accept the results of calibration or repeat process again
        ACCEPT_CALIBRATION = int(input(f'\n\nCALIBRATION STATUS: {calibration_status}. \n\nPRESS 1 TO CONTINUE. PRESS 0 TO REPEAT.\n\n'))
        if ACCEPT_CALIBRATION:
            calibration_result = calibration.compute_and_apply()

            # plot calibration results?
            calibrated_results = calibration_result.calibration_points

            # leave calibration mode
            calibration.leave_calibration_mode()
            break

elif ENTER_CALIBRATION == 2:
    pass

else:
    sys.exit()

## ========================================================
## Validate
## ========================================================
ENTER_VALIDATION = int(input('\n\n PRESS 1 TO ENTER VALIDATION. \n PRESS 2 TO SKIP \n PRESS 0 TO EXIT. \n\n'))

if ENTER_VALIDATION == 1:

    #validation_status = [False, False, False, False, False]
    points_to_validate = [Point2(0.2, 0.2), # upper left
                          Point2(0.8, 0.2), #upper right
                          Point2(0.5, 0.5), # middle
                          Point2(0.2, 0.8), # lower left
                          Point2(0.8, 0.8)] # lower right
    point_names = ['Top Left', 'Top Right', 'Middle', 'Bottom Left', 'Bottom Right']
    sample_count = 30
    timeout_ms = 1000
    validation = ScreenBasedCalibrationValidation(my_et, sample_count, timeout_ms)
    validation.enter_validation_mode()

    while True:

        # cycle through points to calibrate on
        for i, point in enumerate(points_to_validate):

            print(f'Validating position {point_names[i]}')
            while True:
                
                validation.start_collecting_data(point)
                while validation.is_collecting_data:
                    time.sleep(1)                
                NEXT_POINT = int(input(f'\nPRESS 1 TO CONTINUE. PRESS 0 TO REPEAT. \n\n'))
                if NEXT_POINT:
                    break
                else:
                    validation.discard_data(point)

        # compute validation result
        validation_result = validation.compute()
        # print output
        print(f'''
              VALIDATION POINTS: {validation_result.points}
              ACCURACY LEFT EYE: {validation_result.average_accuracy_left}
              ACCURACY RIGHT EYE: {validation_result.average_accuracy_right}
              PRECISION LEFT EYE: {validation_result.average_precision_left}
              PRECISION RIGHT EYE: {validation_result.average_precision_right}
              ''')

        # ask if user wants to accept the results of validation or repeat process again
        ACCEPT_VALIDATION = int(input(f'\n\nPRESS 1 TO CONTINUE. PRESS 0 TO REPEAT.\n\n'))
        if ACCEPT_VALIDATION:

            # leave calibration mode
            validation.leave_validation_mode()
            break
        
elif ENTER_VALIDATION == 2:
    pass

else:
    sys.exit()

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
    global winsize # adjust for differnet coordinates - check this!! Should this be 960 x 520 to account for PROPixx QUAD4X?

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
def write_buffer_to_file(gaze_data_buffer, output_path): 

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

    # save data automatically
    if trig == 254:
        FILENAME = f'{SUBID}_task-{EXPERIMENT}_eye_block-{block}.csv'
        FILEPATH = DATAPATH + SUBID + '/eye/' + FILENAME
        time.sleep(1) # add a bit of extra time at the end so epochs in analysis aren't dropped due to being too short
        write_buffer_to_file(gaze_data_buffer, FILEPATH) # external signal buffer
        print('Saving data')
        block += 1 
    if trig == 255:
        write_buffer_to_file(gaze_data_buffer, FILEPATH)
        print('Exiting')
        break

# stop recording
my_et.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
my_et.unsubscribe_from(tr.EYETRACKER_EXTERNAL_SIGNAL, external_signal_callback)
print('ENDING RECORDING')

