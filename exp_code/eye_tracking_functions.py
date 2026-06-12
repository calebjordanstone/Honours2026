import tobii_research as tr
from tobii_research_addons import ScreenBasedCalibrationValidation, Point2
import pandas as pd
import os
import time
import sys

## ========================================================
## Calibrate
## ========================================================
accepted_inputs = ['0', '1']
def enter_calibration(user_input, eye_tracker):

    if user_input == 1:

        calibration_status = [False, False, False, False, False]
        points_to_calibrate = [(0.25, 0.25), # upper left
                            (0.75, 0.25), #upper right
                            (0.5, 0.5), # middle
                            (0.25, 0.75), # lower left
                            (0.75, 0.75)] # lower right
        point_names = ['Top Left', 'Top Right', 'Middle', 'Bottom Left', 'Bottom Right']
            
        calibration = tr.ScreenBasedCalibration(eye_tracker)
        calibration.enter_calibration_mode()

        while True:

            # cycle through points to calibrate on
            for i, point in enumerate(points_to_calibrate):

                print(f'Calibrating position {point_names[i]}')
                while True:
                    STATUS = calibration.collect_data(point[0], point[1])
                    if STATUS == 'calibration_status_success':
                        calibration_status[i] = True
                    while True:
                        NEXT_POINT = input(f'\nCALIBRATION POINT STATUS: {STATUS} \n\n PRESS 1 TO CONTINUE. PRESS 0 TO REPEAT. \n\n')
                        if NEXT_POINT in accepted_inputs:
                            NEXT_POINT = int(NEXT_POINT)
                            break
                        else:
                            print('Input not valid! Try again')
                    if NEXT_POINT == 1:
                         break
                    else:
                        calibration.discard_data(point[0], point[1])

            # ask if user wants to accept the results of calibration or repeat process again
            while True:
                ACCEPT_CALIBRATION = input(f'\n\nCALIBRATION STATUS: {calibration_status}. \n\nPRESS 1 TO CONTINUE. PRESS 0 TO REPEAT.\n\n')
                if ACCEPT_CALIBRATION in accepted_inputs:
                    ACCEPT_CALIBRATION = int(ACCEPT_CALIBRATION)
                    break
                else:
                    print('Input not valid! Try again')

            if ACCEPT_CALIBRATION:
                calibration_result = calibration.compute_and_apply()

                # plot calibration results?
                calibrated_results = calibration_result.calibration_points

                # leave calibration mode
                calibration.leave_calibration_mode()
                break

    elif user_input == 2:
        pass

    else:
        sys.exit()

## ========================================================
## Validate
## ========================================================
def enter_validation(user_input, eye_tracker):

    if user_input == 1:

        #validation_status = [False, False, False, False, False]
        points_to_validate = [Point2(0.25, 0.25), # upper left
                              Point2(0.75, 0.25), #upper right
                              Point2(0.5, 0.5), # middle
                              Point2(0.25, 0.75), # lower left
                              Point2(0.75, 0.75)] # lower right
        point_names = ['Top Left', 'Top Right', 'Middle', 'Bottom Left', 'Bottom Right']
        sample_count = 30
        timeout_ms = 1000
        validation = ScreenBasedCalibrationValidation(eye_tracker, sample_count, timeout_ms)
        validation.enter_validation_mode()

        while True:

            # cycle through points to calibrate on
            for i, point in enumerate(points_to_validate):

                print(f'Validating position {point_names[i]}')
                while True:
                    
                    validation.start_collecting_data(point)
                    while validation.is_collecting_data:
                        time.sleep(1)
                    while True:
                        NEXT_POINT = input(f'\nPRESS 1 TO CONTINUE. PRESS 0 TO REPEAT. \n\n')
                        if NEXT_POINT in accepted_inputs:
                            NEXT_POINT = int(NEXT_POINT)
                            break
                        else:
                            print('Input not valid! Try again')

                    if NEXT_POINT == 1:
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
            while True:
                ACCEPT_VALIDATION = input(f'\n\nPRESS 1 TO CONTINUE. PRESS 0 TO REPEAT.\n\n')
                if ACCEPT_VALIDATION in accepted_inputs:
                    ACCEPT_VALIDATION = int(ACCEPT_VALIDATION)
                    break
                else:
                    print('Input not valid! Try again')
            
            if ACCEPT_VALIDATION == 1:

                # leave calibration mode
                validation.leave_validation_mode()
                break
            else:
                for point in points_to_validate:
                    validation.discard_data(point)
            
    elif user_input == 2:
        pass

    else:
        sys.exit()