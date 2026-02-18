from datetime import datetime
from psychopy import core, event, visual, logging, gui, event, monitors, data 
#import psychopy_visionscience
from pathlib import Path
from propixx_functions import *
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import json
#from pypixxlib import _libdpx as dp
import Honours2026_ppx_QUAD4X_practice
logging.console.setLevel(logging.WARNING)
## ======================================================================
## Get experiment and monitor information 
## ======================================================================
expInfo = {
    'Monitor Name': 'test',
    'Monitor Refresh Rate': '120',
    'Subject Number': '',
    'Screen Number': '1',
    'Session': 'main',
    'Full Screen': True,
    'Practice': True,
    'Demographics': True,
    'Debrief': True}

dlg = gui.DlgFromDict(
    expInfo, 
    title = 'Experinent Information')

if not dlg.OK:
    core.quit()

EXPERIMENT = 'Honours2026'
DATETIME = datetime.now().strftime("%y%m%d%H%M") 
SUBID = 'sub-' + f'{int(expInfo["Subject Number"])}'.zfill(2)
REFRATE = int(expInfo['Monitor Refresh Rate'])
SCREEN = int(expInfo['Screen Number'])
FULLSCREEN = expInfo['Full Screen']
PRACTICE = expInfo['Practice']
DEMOGRAPHICS = expInfo['Demographics']
DEBRIEF = expInfo['Debrief']

# configure monitor
if expInfo['Monitor Name'] in monitors.getAllMonitors():
    MONITOR = monitors.Monitor(expInfo['Monitor Name'])

else:
    monInfo = {
        'Monitor y (pxl)': '1080',
        'Monitor x (pxl)': '1920',
        'Monitor Width (cm)': '',
        'Monitor Distance (cm)': ''}
    dlg = gui.DlgFromDict(
        monInfo, 
        title = 'Monitor Information')

    if dlg.OK:
        MONITOR = monitors.Monitor(expInfo['Monitor Name'])
        MONITOR.setDistance(int(monInfo['Monitor Distance (cm)']))
        MONITOR.setWidth(float(monInfo['Monitor Width (cm)']))
        MONITOR.setSizePix((int(monInfo['Monitor x (pxl)']), 
                            int(monInfo['Monitor y (pxl)'])))
        MONITOR.saveMon()
    else:
        gui.criticalDlg('Something went wrong. Aborting')
        core.quit()

# collect demographics
if DEMOGRAPHICS:

    demographics = {
        'How old are you?': '',
        'How do you describe your sex?': ['Female', 'Male', 'Prefer not to say', 'Intersex', 'Other'],
        'Which is your dominant hand?': ['Right', 'Left', 'Ambidextrous'],
        'Do you have normal or corrected-to-normal hearing?': False,
        'Do you have normal or corrected-to-normal vision?': False,
        'Do you have a history of photosensitivity, epilepsy, or migraines/headaches?': True}

    if expInfo['Demographics']:

        dlg = gui.DlgFromDict(
            demographics, 
            title = 'Demographics and Screening')
        if not dlg.OK:
            core.quit()

    AGE = int(demographics['How old are you?'])
    SEX = demographics['How do you describe your sex?']
    HAND = demographics['Which is your dominant hand?']
    HEARING = demographics['Do you have normal or corrected-to-normal hearing?']
    VISION = demographics['Do you have normal or corrected-to-normal vision?']
    PHOTOSENSITIVE = demographics['Do you have a history of photosensitivity, epilepsy, or migraines/headaches?']

    # check eligibitiy 
    if (PHOTOSENSITIVE) or (not HEARING) or (not VISION) or (AGE <17):
        print(f'Eligibility criteria not met. Please contact the experimenter.\n\nPHOTOSENSITIVE = {PHOTOSENSITIVE}\nVISION = {VISION}\nHEARING = {HEARING}\nAGE = {AGE}')
        core.quit()

## ======================================================================
## Create data paths
## ======================================================================
DATAPATH = 'C:/Users/cstone/OneDrive - UNSW/Documents/Projects/honours_projects/Honours2026/data/'
# DATAPATH = '/home/experimenter/Experiments/Honours2026/data/'
os.mkdir(DATAPATH + SUBID)
folders = ['/beh/', '/eeg/']
for folder in folders:
    os.mkdir(DATAPATH + SUBID + folder)
FILENAME = f'{SUBID}_task-{EXPERIMENT}_beh.txt'
LOGFILENAME = f'{SUBID}_task-{EXPERIMENT}_log.txt'
FRMSFILENAME = f'{SUBID}_task-{EXPERIMENT}_frms.txt'
FILEPATH = DATAPATH + SUBID + folders[0] + FILENAME
LOGFILEPATH = DATAPATH + SUBID + folders[0] + LOGFILENAME
FRMSFILEPATH = DATAPATH + SUBID + folders[0] + FRMSFILENAME
logging.LogFile(LOGFILEPATH)
## ======================================================================
## Create window, stimuli, and other experiment features
## ======================================================================
# create window
win = visual.Window(
        screen = SCREEN, 
        monitor = MONITOR,
        size = MONITOR.getSizePix(),
        fullscr=FULLSCREEN,
        pos = [0, 0],
        color = [0, 0, 0], # black
        units = 'pix', 
        colorSpace = 'rgb255',
        blendMode = 'avg')
win.refreshThreshold = 1/REFRATE + 0.002 ## is this still appropriate with quad4x?

# set mouse to invisible
mouse = event.Mouse(visible=False)

# create clock
clock = core.Clock()

# define stimuli positions
size_params = { # common parameters used to calculate stimulus size in pixels
    'distance': MONITOR.getDistance(), 
    'screen_res': MONITOR.getSizePix(), 
    'screen_width': MONITOR.getWidth()
}
# define positions for tagged stimuli
y_pos = dva_to_pix(4, **size_params) # stmuli to appear 4 degrees up/down from centre # check Ferrante et al. paper
x_pos = dva_to_pix(6, **size_params) # stmuli to appear 4 degrees to left/right of centre
LEFT = [-x_pos, -y_pos]
RIGHT = [x_pos, -y_pos]
# define positions for text stimuli
CENTRE = [0, 0]
CENTRE_HIGH = [0, y_pos]
# define positions for left/right scale stimuli
scale_y = dva_to_pix(2, **size_params)
scale_length = dva_to_pix(10, **size_params)
SCALE_START_LOW = [-scale_length/2, -scale_y]
SCALE_END_LOW = [scale_length/2, -scale_y]
# define positions for conf scale stimuli
conf_x = dva_to_pix(3, **size_params)
SCALE_UP_LEFT = [-conf_x, scale_length/2]
SCALE_DOWN_LEFT = [-conf_x, -scale_length/2]
# define trigger position
TLC = [-win.size[0]/2, win.size[1]/2] # top left corner

# create new positions to display stimuli in each quadrant
CENTRE_QUAD = reformat_for_propixx(win, CENTRE) 
CENTRE_HIGH_QUAD = reformat_for_propixx(win, CENTRE_HIGH)
LEFT_QUAD = reformat_for_propixx(win, LEFT)
RIGHT_QUAD = reformat_for_propixx(win, RIGHT)
TLC_QUAD = reformat_for_propixx(win, TLC)
SCALE_START_QUAD = reformat_for_propixx(win, SCALE_START_LOW)
SCALE_END_QUAD = reformat_for_propixx(win, SCALE_END_LOW)
SCALE_UP_LEFT_QUAD = reformat_for_propixx(win, SCALE_UP_LEFT) 
SCALE_DOWN_LEFT_QUAD =  reformat_for_propixx(win, SCALE_DOWN_LEFT) 

# define colours
# BLUE =   [37, 141, 165] 
ORANGE = [194, 99, 32] 
WHITE = [255, 255, 255]
GREY = [80, 80, 80]
DARK_GREY = [50, 50, 50]
BLACK = [0, 0, 0]

# define response keys
RESPKEYS = ['c', 'm']
QUITKEYS = ['escape']
CONKEYS = ['space']
EXITKEYS = ['enter']

# define number of trials/blocks
N_PRACTICE_TRIALS = 32
N_TRIALS = 64 
N_BLOCKS = 20
N_STATES = 8
N_STATE_CHANGES = N_STATES - 1
N_TRIALS_PER_STATE = int((N_TRIALS * N_BLOCKS) / N_STATES)
N_TRIALS_CONF = 8

# define target ratios
HIGH = [0.9, 0.1]
MED_HIGH = [0.8, 0.2]
MED_LOW = [0.7, 0.3]
LOW = [0.6, 0.4]

# define points
POINTS_COR = 10

# calculate total possible points
TOTAL_POINTS = N_TRIALS*N_BLOCKS*POINTS_COR
TOTAL_REWARD = 1000 # $15, or 1500 cents
CENTS_PER_POINT = TOTAL_REWARD / TOTAL_POINTS

# define timing (multiply all by 4 to account for increase in refrate from PROPixx)
ITI_RANGE = [int(0.5*REFRATE)*4, int(0.8*REFRATE)*4] 
# TAG_DUR = int(0.3*REFRATE)*4 # initial invisible tag for 0.2 seconds
PLH_DUR = int(1*REFRATE)*4 # placeholder on screen for 0.2 seconds
TAR_DUR = int(0.067*REFRATE)*4 
MSK_DUR = int(0.067*REFRATE)*4
RSP_DUR = int(0.866*REFRATE)*4
FDB_DUR = int(0.4*REFRATE)*4

# create oscillation for stimuli
framerate = REFRATE * 4
duration = 3 # seconds
t = np.tile(np.linspace(0, 1, framerate, endpoint=False), duration) 
fs = [60, 64] #[framerate/8, framerate/7] # set frequencies 
a = 0.5 # ampltiude
p_60 = 0 # phase
p_64 = 0 # phase
opacity_60_Hz = a*(np.sin(2*np.pi*fs[0]*t + p_60)) + 0.5
opacity_64_Hz = a*(np.sin(2*np.pi*fs[1]*t + p_64)) + 0.5

# create stimuli for each trial
stim_params = { # common parameters used across stimuli
    'win': win, 
    'units': 'pix', 
    'opacity': 1,
    'contrast': 1,
    'colorSpace': 'rgb255'}
txt_stim_height = dva_to_pix(dva=1, **size_params)
txt_stim = visual.TextStim(
    pos=CENTRE,
    color=WHITE,
    height=txt_stim_height,
    wrapWidth=MONITOR.getSizePix()[0]/2*0.8, # 80% of width of screen
    **stim_params)
fix_stim_height = dva_to_pix(dva=1, **size_params)
fix_stim = visual.TextStim(
    text= "+",
    pos=CENTRE,
    color=WHITE,
    height=fix_stim_height,
    **stim_params)
trig_stim = visual.Rect(
    width=1, 
    height=1,
    lineColor=None, 
    interpolate=False,
    **stim_params)

# note: for the below stimuli, suffix "1" denotes left and suffix "2" denotes right side of the screen
crc_stim_rad = dva_to_pix(dva=3, **size_params)
plh_stim1 = visual.Circle(
    lineColor = WHITE,
    fillColor = DARK_GREY,
    radius = crc_stim_rad,
    edges = 100,
    lineWidth = 1,
    **stim_params)
plh_stim2 = visual.Circle(
    lineColor = WHITE,
    fillColor = DARK_GREY,
    radius = crc_stim_rad,
    edges = 100,
    lineWidth = 1,
    **stim_params)
grt_stim1 = visual.GratingStim(
    tex = 'sin',
    mask = 'gauss',
    maskParams = {"sd": 3.8},
    sf = .05,
    size = crc_stim_rad*2,
    **stim_params)
grt_stim2 = visual.GratingStim(
    tex = 'sin',
    mask = 'gauss',
    maskParams = {"sd": 3.8},
    sf = .05,
    size = crc_stim_rad*2,
    **stim_params)
msk_stim1 = visual.GratingStim(
    mask="gauss",
    maskParams = {"sd": 3.8},
    size=crc_stim_rad*2, 
    interpolate=False, 
    autoLog=False, 
    **stim_params)
msk_stim2 = visual.GratingStim(
    mask="gauss",
    maskParams = {"sd": 3.8},
    size=crc_stim_rad*2, 
    interpolate=False, 
    autoLog=False, 
    **stim_params)

# create stimuli for probe questions
probe_txt_stim = visual.TextStim(
    pos=CENTRE_HIGH,
    color=WHITE,
    height=txt_stim_height,
    wrapWidth=MONITOR.getSizePix()[0]/2*0.8, # 80% of width of screen
    **stim_params)
left_txt_stim = visual.TextStim(
                            text = 'Left', 
                            color = WHITE,
                            height = txt_stim_height*2, 
                            **stim_params)
right_txt_stim = visual.TextStim(
                             text = 'Right', 
                             color = WHITE,
                             height = txt_stim_height*2, 
                             **stim_params)
scale_prob = visual.Rect(
    color=WHITE,
    size=(scale_length, scale_length/12),
    **stim_params)
fifty_txt_stim = visual.TextStim(
                            text = '50%', 
                            color = WHITE,
                            height = txt_stim_height, 
                            **stim_params)
hundred_txt_stim = visual.TextStim(
                            text = '100%', 
                            color = WHITE,
                            height = txt_stim_height, 
                            **stim_params)
scale_conf = visual.Rect(
    color=WHITE,
    size=(scale_length/12, scale_length),
    **stim_params)
guess_txt_stim = visual.TextStim(
                            text = 'Guess', 
                            color = WHITE,
                            height = txt_stim_height, 
                            **stim_params)
certain_txt_stim = visual.TextStim(
                            text = 'Certain', 
                            color = WHITE,
                            height = txt_stim_height, 
                            **stim_params)
mkr_stim_rad = dva_to_pix(dva=0.5, **size_params)
marker_stim = visual.Circle(
    edges=100,
    radius=mkr_stim_rad,
    **stim_params)
marker_stim.color = ORANGE

# create list of all stimuli to use later ## Add trig stim?
stims = [txt_stim, 
         fix_stim,
         #tag_stim1, tag_stim2,
         plh_stim1, plh_stim2,
         grt_stim1, grt_stim2, 
         msk_stim1, msk_stim2, 
         probe_txt_stim,
         left_txt_stim, right_txt_stim,
         #scale_left, scale_right,
         #fifty_left_txt_stim, hundred_left_txt_stim,
         #fifty_right_txt_stim, hundred_right_txt_stim,
         scale_prob, scale_conf,
         marker_stim,
         guess_txt_stim, certain_txt_stim,
         fifty_txt_stim, hundred_txt_stim] 

# rescale stimuli to half size to account for resolution drop
for stim in stims:
    stim.size = stim.size/2 

# create experiment handler to save output
if not DEMOGRAPHICS:
    AGE = np.nan
    SEX = np.nan
    HAND = np.nan

experimentDict = {
    'Experiment': EXPERIMENT,
    'Date': DATETIME,
    'Refrate': REFRATE,
    'MonitorSize' : MONITOR.getSizePix(),
    'Subject': SUBID,
    'SubjectAge': AGE,
    'SubjectSex': SEX,
    'SubjectHandedness': HAND}

exp = data.ExperimentHandler(name=EXPERIMENT,
                             extraInfo=experimentDict)
exp.dataNames = []                     

# set placeholder values for metacognitive questions
choice = np.nan 
confidence = np.nan
probability = np.nan
choice_rt = np.nan
confidence_rt = np.nan
probability_rt = np.nan

# write instruction text
instructText = {

'INSTRUCTIONS_1': '''
    ## INSERT ##
    ''',

'INSTRUCTIONS_2': '''
    ## INSERT ##
    ''',

'INSTRUCTIONS_3': '''
    ## INSERT ##
    ''',

'INSTRUCTIONS_4': '''
    ## INSERT ##
    '''
}
## ======================================================================
## Create trial structure
## ======================================================================
# define the possible trial types when the target is on the left
stim_tag_combos_tar_left = np.repeat(
    np.asarray(
        tuple(
            itertools.product(
                [[-45, 0], [-45, 90],
                 [45, 0], [45, 90]],
                [(60, 64), (64, 60)])),
            dtype = 'object'), 
    repeats=1,
    axis=0)
# define the possible trial types when the target is on the right
stim_tag_combos_tar_right = np.repeat(
    np.asarray(
        tuple(
            itertools.product(
                [[0, -45], [0, 45], 
                 [90, -45], [90, 45]],
                [(60, 64), (64, 60)])),
            dtype = 'object'), 
    repeats=1,
    axis=0)
# create the trial structure for target assignment and ratios 
trial_structure = np.repeat(
    np.asarray(
        tuple(
            itertools.product(
                [HIGH, MED_HIGH, MED_LOW, LOW], # target ratio  
                ['left', 'right'])), # HPL  
            dtype = 'object'), 
    repeats=1,
    axis=0)
np.random.shuffle(trial_structure)
# combine the above to get the trial sequence for the experiment
exp_target_ratio = []
exp_hpl = []
exp_trial_seq = []
exp_target_side = []
for state in trial_structure:
    # get ratio and hpl information for the current state
    ratio = state[0]
    hpl = state[1]
    # define hpl and lpl
    if hpl == 'right':
        lpl = 'left'
    else: 
        lpl = 'right'
    # get number of trials with target in the HPL and target in the LPL
    n_trials_high = ratio[0] * N_TRIALS_PER_STATE
    n_trials_low = ratio[1] * N_TRIALS_PER_STATE
    if hpl == 'right':
        state_trial_seq = np.concatenate(
            [np.repeat(stim_tag_combos_tar_right, repeats=int(n_trials_high/8), axis=0), # repeats = number of trials needed, divided by the number of trial types with the target on the HPL side
             np.repeat(stim_tag_combos_tar_left, repeats=int(n_trials_low/8), axis=0)]
            )
    elif hpl == 'left':
        state_trial_seq = np.concatenate(
            [np.repeat(stim_tag_combos_tar_left, repeats=int(n_trials_high/8), axis=0), # repeats = number of trials needed, divided by the number of trial types with the target on the HPL side
             np.repeat(stim_tag_combos_tar_right, repeats=int(n_trials_low/8), axis=0)]
            )
    np.random.shuffle(state_trial_seq)
    # find target side on each trial
    for i, _ in enumerate(state_trial_seq):
        if state_trial_seq[i][0][0] in [-45, 45]:
            target_side = ['left']
        else:
            target_side = ['right']
        exp_target_side.extend(target_side)
    # add to whole experiment array
    exp_target_ratio.extend([ratio]*N_TRIALS_PER_STATE)
    exp_hpl.extend([hpl]*N_TRIALS_PER_STATE)
    exp_trial_seq.extend(state_trial_seq)

# create random phase offsets
phase_offsets = np.repeat(np.arange(1, 9), repeats=(N_TRIALS*N_BLOCKS)/8) # offset phase by 1-8 samples from the cycle
np.random.shuffle(phase_offsets)

# define trigger values
'''
Block number triggers:
100 + block number. e.g., 101 = block 1

Trial number triggers:
1 to N_TRIALS. e.g., 1 = trial 1

Trial structure triggers:
Ratio: H = high, MH = medium high, ML = medium low, L = low
High-probability location: L = left, R = right
Target location: l = left, r = right
HLl: 111xx
HLr: 112xx
HRl: 121xx
HRr: 122xx

MHLl: 211xx
MHLr: 212xx
MHRl: 221xx
MHRr: 222xx

MLLl: 311xx
MLLr: 312xx
MLRl: 321xx
MLRr: 322xx

LLl: 411xx
LLr: 412xx
LRl: 421xx
LRr: 422xx

ITI: xxx00 # inter-trial interval onset
PLH: xxx10 # placeholder onsest
TAR: xxx20 # target onset
MSK: xxx30 # mask onset
RES: xxx40 # correct response
RES: xxx50 # incorrect response
RES: xxx60 # missed response

Probe Questions:
choice: 500
confidence: 600
probability: 700
'''

def find_trigger_prefix(trl_idx, exp_target_ratio, exp_hpl, exp_target_side):

    # find prefix for state ratio for current trial
    if exp_target_ratio[trl_idx] == HIGH:
        pfx_1 = 1
    elif exp_target_ratio[trl_idx] == MED_HIGH:
        pfx_1 = 2
    elif exp_target_ratio[trl_idx] == MED_LOW:
        pfx_1 = 3
    else: 
        pfx_1 = 4
    # find prefix for HPL side for current trial
    if exp_hpl == 'left':
        pfx_2 = 1
    else:
        pfx_2 = 2
    # find prefix for target side for current trial
    if exp_target_side[trl_idx] == 'left':
        pfx_3 = 1
    else: 
        pfx_3 = 2
    # add zeros to end
    sfx = '00'
    # combine into one string
    trig = f"{pfx_1}{pfx_2}{pfx_3}{sfx}"
    # return as integer
    return int(trig)

## ======================================================================
## Initialise PROPixx
## ======================================================================
# establish connection to hardware
dp.DPxOpen()
isReady = dp.DPxIsReady()
if isReady:
    dp.DPxSetPPxDlpSeqPgrm('QUAD4X') # set to 4x refresh rate
    dp.DPxEnableDoutPixelMode() # enable pixel mode for triggers
    dp.DPxEnablePPxRearProjection() # enable rear projection to reverse display
    dp.DPxWriteRegCache()
else:
    print('Warning! DPx call failed, check connection to hardware')
    core.quit()

## ======================================================================
## Present stimuli
## ======================================================================
## Present instructions 
for txt in instructText.keys():

    event.clearEvents()
    logging.warning(f'{txt}')
    logging.flush()
    txt_stim.text = instructText[txt]
    instr_idx = 0
    while True:

        # keep track of quadrants
        quad_idx = (instr_idx % 4)

        # draw
        txt_stim.pos = CENTRE_QUAD[quad_idx]
        txt_stim.draw()

        # collect user input to exit 
        pressed = event.getKeys(keyList = CONKEYS)
        if pressed: 
            break

        # flip window once fourth quadrant is drawn
        if quad_idx == 3: 
            win.flip()

        # udpate frame
        instr_idx += 1 

## Start practice trials 
logging.warning('START_PRAC')
logging.flush()
while True: 

    # present instructions 
    txt_stim.text = f'This is a practice block. \n\n Please keep your eyes on the fixation cross throughout each trial. \n\n Press space to begin.' 
    txt_idx = 0
    while True:

        # keep track of quadrants
        quad_idx = (txt_idx % 4)

        # draw
        txt_stim.pos = CENTRE_QUAD[quad_idx]
        txt_stim.draw()

        # collect user input to exit 
        pressed = event.getKeys(keyList = CONKEYS)
        if pressed:
            break

        # flip window once fourth quadrant is drawn
        if quad_idx == 3: 
            win.flip()

        # udpate frame
        txt_idx += 1 
    
    # start presenting trials ---------------------------------------------------------- TRIAL ONSET
    for trial in range(0, N_PRACTICE_TRIALS):

        # present confidence rating screens after ever N_TRIALS_CONF trials
        if (trial > 0) & (trial % N_TRIALS_CONF == 0):

            mouse.setVisible(True)
            event.clearEvents()
            probe_txt_stim.text = "On which side do you think the target appears most often?"
            txt_idx = 0
            while True: # --------------------------------------------------- LEFT/RIGHT ONSET

                # keep track of quadrants
                quad_idx = (txt_idx % 4)

                # set positions
                probe_txt_stim.pos = CENTRE_HIGH_QUAD[quad_idx]
                left_txt_stim.pos = LEFT_QUAD[quad_idx]
                right_txt_stim.pos = RIGHT_QUAD[quad_idx]

                # draw
                probe_txt_stim.draw()
                left_txt_stim.draw()
                right_txt_stim.draw()

                # collect response input
                buttons, times = mouse.getPressed(getTime=True)
                if buttons[0]: #if mouse gets pressed
                    mousePos = mouse.getPos() # get mouse position
                    choice_rt = times[0]
                    if left_txt_stim.contains(mousePos):
                        choice = "left"
                        break
                    elif right_txt_stim.contains(mousePos):
                        choice = "right"
                        break

                # flip window once fourth quadrant is drawn
                if quad_idx == 3: 
                    win.flip()

                # udpate frame
                txt_idx += 1 
            
            # reset everything
            core.wait(0.15)
            event.clearEvents()
            mouse.clickReset()
            mouse.setVisible(False)
            probe_txt_stim.text = f"How confident are you that the target appears on the {choice.upper()} side most often?"
            txt_idx = 0
            while True: # --------------------------------------------------- CONF ONSET

                # keep track of quadrants
                quad_idx = (txt_idx % 4)

                # set positions
                probe_txt_stim.pos = CENTRE_HIGH_QUAD[quad_idx]
                scale_prob.pos = CENTRE_QUAD[quad_idx]
                guess_txt_stim.pos = SCALE_START_QUAD[quad_idx]
                certain_txt_stim.pos = SCALE_END_QUAD[quad_idx]

                # set marker location
                mousePos = mouse.getPos()
                mousePos_reformated = reformat_for_propixx(win, [mousePos[0], mousePos[1]])

                if mousePos_reformated[quad_idx][0] <= SCALE_START_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = SCALE_START_QUAD[quad_idx][0]
                if mousePos_reformated[quad_idx][0] >= SCALE_END_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = SCALE_END_QUAD[quad_idx][0]
                mousePos_reformated[quad_idx][1] = CENTRE_QUAD[quad_idx][1]
                marker_stim.pos = mousePos_reformated[quad_idx]

                # collect response
                buttons, times = mouse.getPressed(getTime=True)
                if buttons[0]: #if mouse gets pressed
                    low = guess_txt_stim.pos[0]
                    high = certain_txt_stim.pos[0]
                    rating = marker_stim.pos[0]
                    confidence = (rating + abs(low)) / (high + abs(low))
                    confidence_rt = times[0]
                    break

                # draw
                probe_txt_stim.draw()
                scale_prob.draw()
                certain_txt_stim.draw()
                guess_txt_stim.draw()
                marker_stim.draw()

                # flip window once fourth quadrant is drawn
                if quad_idx == 3: 
                    win.flip()

                # udpate frame
                txt_idx += 1 
    
            # update probe text
            probe_txt_stim.text = f"On what percentage of trials do you think the target appears on the {choice.upper()} side?"
            
            # reset everything
            event.clearEvents()
            mouse.clickReset()
            core.wait(0.15)  
            txt_idx = 0
            while True: # --------------------------------------------------- PROB ONSET

                # keep track of quadrants
                quad_idx = (txt_idx % 4)

                # set positions
                probe_txt_stim.pos = CENTRE_HIGH_QUAD[quad_idx]
                scale_prob.pos = CENTRE_QUAD[quad_idx]
                fifty_txt_stim.pos = SCALE_START_QUAD[quad_idx]
                hundred_txt_stim.pos = SCALE_END_QUAD[quad_idx]

                # set marker location
                mousePos = mouse.getPos()
                mousePos_reformated = reformat_for_propixx(win, [mousePos[0], mousePos[1]])

                if mousePos_reformated[quad_idx][0] <= SCALE_START_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = SCALE_START_QUAD[quad_idx][0]
                if mousePos_reformated[quad_idx][0] >= SCALE_END_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = SCALE_END_QUAD[quad_idx][0]
                mousePos_reformated[quad_idx][1] = CENTRE_QUAD[quad_idx][1]
                marker_stim.pos = mousePos_reformated[quad_idx]

                # collect response
                buttons, times = mouse.getPressed(getTime=True)
                if buttons[0]: #if mouse gets pressed
                    zero = fifty_txt_stim.pos[0]
                    one = hundred_txt_stim.pos[0]
                    rating = marker_stim.pos[0]
                    probability = (rating + abs(zero)) / (one + abs(zero))
                    probability_rt = times[0]
                    break

                # draw
                probe_txt_stim.draw()
                scale_prob.draw()
                fifty_txt_stim.draw()
                hundred_txt_stim.draw()
                marker_stim.draw()
                
                # flip window once fourth quadrant is drawn
                if quad_idx == 3: 
                    win.flip()

                # udpate frame
                txt_idx += 1 

        # reset everything
        event.clearEvents()
        mouse.clickReset()
        mouse.setVisible(False)
        core.wait(0.5) # pause to let participants find response buttongs again   
     
        # set trial properites
        stim_options = [[-45, 0], [-45, 90], [45, 0], [45, 90], 
                        [0, -45], [0, 45], [90, -45], [90, 45]]
        stim_idx = np.random.choice(8)
        grt_stim1.ori = stim_options[stim_idx][0]
        grt_stim2.ori = stim_options[stim_idx][1]

        # set tag properties
        tag_idx = 0
        freq_options = [[60, 64], [64, 60]]
        freq_idx = np.random.choice(2)
        if freq_options[freq_idx][0] == 60:
            freq1 = opacity_60_Hz
            freq2 = opacity_64_Hz 
        elif freq_options[freq_idx][0] == 64:
            freq1 = opacity_64_Hz 
            freq2 = opacity_60_Hz

        # create fresh mask texture each trial
        noiseTexture = np.random.rand(32, 32) * 2.0 - 1
        msk_stim1.tex=noiseTexture
        msk_stim2.tex=noiseTexture

        # set random ITI duration
        ITI_DUR = np.random.randint(ITI_RANGE[0], ITI_RANGE[1])

        event.clearEvents()
        # start stimulus presentation -------------------------------------------------- ITI ONSET
        for frame in range(0, ITI_DUR): 

            if frame == 0:
                logging.warning('START_ISI')
                logging.flush()

            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw fixation
            fix_stim.pos = CENTRE_QUAD[quad_idx]
            fix_stim.draw()

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

        for frame in range(0, PLH_DUR): # ------------------------------------------------ PLACEHOLDER ONSET

            if frame == 0:
                logging.warning('START_PHD')
                logging.flush()
            
            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw fixation
            fix_stim.pos = CENTRE_QUAD[quad_idx]
            fix_stim.draw()

            # set placeholder positions
            plh_stim1.pos = LEFT_QUAD[quad_idx]
            plh_stim2.pos = RIGHT_QUAD[quad_idx]

            # set placeholder opacity
            #plh_stim1.opacity = freq1[tag_idx]
            #plh_stim2.opacity = freq2[tag_idx]

            # draw placeholders
            plh_stim1.draw()
            plh_stim2.draw()

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()
            
            # update tags
            tag_idx += 1

        # reset things before target display
        event.clearEvents()
        clock.reset()
        for frame in range(0, TAR_DUR): # ------------------------------------------------ TARGET ONSET

            if frame == 0:
                logging.warning('START_TAR')
                logging.flush()

            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw fixation
            fix_stim.pos = CENTRE_QUAD[quad_idx]
            fix_stim.draw()

            # set placeholder positions
            plh_stim1.pos = LEFT_QUAD[quad_idx]
            plh_stim2.pos = RIGHT_QUAD[quad_idx]
            
            # set placeholder opacity
            #plh_stim1.opacity = freq1[tag_idx]
            #plh_stim2.opacity = freq2[tag_idx]
            
            # set grating positions
            grt_stim1.pos = LEFT_QUAD[quad_idx]
            grt_stim2.pos = RIGHT_QUAD[quad_idx]

            # set grating opacity
            #grt_stim1.opacity = freq1[tag_idx]
            #grt_stim2.opacity = freq2[tag_idx]

            # draw placeholders
            plh_stim1.draw()
            plh_stim2.draw()

            # draw gratings
            grt_stim1.draw()
            grt_stim2.draw()

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

            # update tags
            tag_idx += 1

        for frame in range(0, MSK_DUR): # ------------------------------------------------ MASK ONSET

            if frame == 0:
                logging.warning('START_MSK')
                logging.flush()

            # keep track of quadrants
            quad_idx = (frame % 4)
            
            # draw fixation
            fix_stim.pos = CENTRE_QUAD[quad_idx]
            fix_stim.draw()

            # set placeholder positions 
            plh_stim1.pos = LEFT_QUAD[quad_idx]
            plh_stim2.pos = RIGHT_QUAD[quad_idx]

            # set mask positions
            msk_stim1.pos = LEFT_QUAD[quad_idx]
            msk_stim2.pos = RIGHT_QUAD[quad_idx]
            
            # set placeholder opacity
            #plh_stim1.opacity = freq1[tag_idx]
            #plh_stim2.opacity = freq2[tag_idx]

            # set mask opacity
            #msk_stim1.opacity = freq1[tag_idx]
            #msk_stim2.opacity = freq2[tag_idx]

            # draw placeholders
            plh_stim1.draw()
            plh_stim2.draw()

            # draw masks
            msk_stim1.draw()
            msk_stim2.draw()

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

            # update tags
            tag_idx += 1

        for frame in range(0, RSP_DUR): # ------------------------------------------------ RESPONSE ONSET

            if frame == 0:
                logging.warning('START_RSP')
                logging.flush()
            
            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw fixation
            fix_stim.pos = CENTRE_QUAD[quad_idx]
            fix_stim.draw()

            # set placeholder positions and fill
            plh_stim1.pos = LEFT_QUAD[quad_idx]
            plh_stim2.pos = RIGHT_QUAD[quad_idx]

            # set placeholder opacity
            #plh_stim1.opacity = freq1[tag_idx]
            #plh_stim2.opacity = freq2[tag_idx]

            # draw placeholders
            plh_stim1.draw()
            plh_stim2.draw()

            # collect response input
            quitPressed = event.getKeys(keyList = QUITKEYS)
            pressed = event.getKeys(keyList = RESPKEYS, timeStamped = clock)
            if quitPressed: # exit task 
                dp.DPxSetPPxDlpSeqPgrm('RGB')
                dp.DPxDisableDoutPixelMode()
                dp.DPxWriteRegCache()
                dp.DPxClose() 
                core.quit()
            elif pressed:
                break

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

            # update tags
            tag_idx += 1
            
        # calculate correct response
        tar_idx = np.isin(stim_options[stim_idx], [45, -45])
        target_ori = stim_options[stim_idx][tar_idx]
        if target_ori == -45:
            correct_response = 'c'
        elif target_ori == 45:
            correct_response = 'm'
        # calculate accuracy and points
        if pressed:
            if response == correct_response:
                    acc = 1
                    points = POINTS_COR
            elif response != correct_response:
                    acc = 0
                    points = 0
            # update feedback text
            if acc == 1:
                txt_stim.text = 'Correct! 10 points.'
            elif acc == 0:
                txt_stim.text = 'Incorrect! No points.'
        elif not pressed:
            response = 999
            rt = 999
            acc = 999
            points = 0
            txt_stim.text = 'Too slow! No points.'
        
        for frame in range(0, FDB_DUR): # --------------------------------------------------- FDB ONSET

            if frame == 0:
                logging.warning('START_FDB')
                logging.flush()

            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw
            txt_stim.pos = CENTRE_QUAD[quad_idx]
            txt_stim.draw()
                
            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

    # present end of block text
    txt_stim.text = f'''
    End of practice block.
    Press space to repeat.''' 
    event.clearEvents()
    txt_idx = 0
    while True:

        # keep track of quadrants
        quad_idx = (txt_idx % 4)

        # draw
        txt_stim.pos = CENTRE_QUAD[quad_idx]
        txt_stim.draw()

        # collect user input to exit 
        continue_practice = event.getKeys(keyList = CONKEYS)
        end_practice = event.getKeys(keyList = EXITKEYS)
        if continue_practice or end_practice:
            break

        # flip window once fourth quadrant is drawn
        if quad_idx == 3: 
            win.flip()

        # udpate frame
        txt_idx += 1

    if end_practice:
        break  

## Present instructions before starting main experiment
event.clearEvents()
txt_stim.text = 'You will now start the main experiment.\nPlease wait for the experimenter to start the EEG recording.'
instr_idx = 0
while True:

    # keep track of quadrants
    quad_idx = (instr_idx % 4)

    # draw
    txt_stim.pos = CENTRE_QUAD[quad_idx]
    txt_stim.draw()

    # collect user input to exit 
    pressed = event.getKeys(keyList = CONKEYS)
    if pressed: 
        break

    # flip window once fourth quadrant is drawn
    if quad_idx == 3: 
        win.flip()

    # udpate frame
    instr_idx += 1 

## Start experiment 
logging.warning('START_EXP')
logging.flush()
framesPerBlock = {}
runningTrialNo = 0
cumulative_points = 0
cumulative_cents = 0
for block in range(1, N_BLOCKS + 1): 

    # present instructions 
    txt_stim.text = f'This is Block {block}. \n\n Please keep your eyes on the fixation cross throughout each trial. \n\n Press space to begin.' 
    txt_idx = 0
    trg_rgb = dp.DPxTriggerToRGB(100 + block) # convert to RGB
    while True:

        # keep track of quadrants
        quad_idx = (txt_idx % 4)

        # draw trigger stimulus
        if txt_idx < 4:
            # trig_stim.lineColor = trg_rgb
            # trig_stim.start = TLC_QUAD[quad_idx]
            # trig_stim.end= [TLC_QUAD[quad_idx][0] + 1, TLC_QUAD[quad_idx][1]]
            trig_stim.color = trg_rgb
            trig_stim.pos = TLC_QUAD[quad_idx]
            trig_stim.draw()

        # draw
        txt_stim.pos = CENTRE_QUAD[quad_idx]
        txt_stim.draw()

        # collect user input to exit 
        pressed = event.getKeys(keyList = CONKEYS)
        if pressed:
            break

        # flip window once fourth quadrant is drawn
        if quad_idx == 3: 
            win.flip()

        # udpate frame
        txt_idx += 1 
    
    # logging 
    logging.warning(f'START_BLOCK_{block}')
    logging.flush()
    # turn on recording of frame intervals
    win.recordFrameIntervals = True
    framesPerTrial = {}
    points_this_block = 0
    cents_this_block = 0
    # start presenting trials ---------------------------------------------------------- TRIAL ONSET
    for trial in range(0, N_TRIALS):

        # present confidence rating screens after ever N_TRIALS_CONF trials
        if (runningTrialNo > 0) & (runningTrialNo % N_TRIALS_CONF == 0):

            logging.warning(f'START_CHOICE')
            logging.flush()
            mouse.setVisible(True)
            event.clearEvents()
            probe_txt_stim.text = "On which side do you think the target appears most often?"
            txt_idx = 0
            trg_rgb = dp.DPxTriggerToRGB(500)
            while True: # --------------------------------------------------- LEFT/RIGHT ONSET

                # keep track of quadrants
                quad_idx = (txt_idx % 4)

                # draw trigger stimulus
                if txt_idx < 4:
                    #trig_stim.lineColor = trg_rgb
                    #trig_stim.start = TLC_QUAD[quad_idx]
                    #trig_stim.end= [TLC_QUAD[quad_idx][0] + 1, TLC_QUAD[quad_idx][1]]
                    trig_stim.color = trg_rgb
                    trig_stim.pos = TLC_QUAD[quad_idx]
                    trig_stim.draw()

                # set positions
                probe_txt_stim.pos = CENTRE_HIGH_QUAD[quad_idx]
                left_txt_stim.pos = LEFT_QUAD[quad_idx]
                right_txt_stim.pos = RIGHT_QUAD[quad_idx]

                # draw
                probe_txt_stim.draw()
                left_txt_stim.draw()
                right_txt_stim.draw()

                # collect response input
                buttons, times = mouse.getPressed(getTime=True)
                if buttons[0]: #if mouse gets pressed
                    mousePos = mouse.getPos() # get mouse position
                    choice_rt = times[0]
                    if left_txt_stim.contains(mousePos):
                        choice = "left"
                        break
                    elif right_txt_stim.contains(mousePos):
                        choice = "right"
                        break

                # flip window once fourth quadrant is drawn
                if quad_idx == 3: 
                    win.flip()

                # udpate frame
                txt_idx += 1 
            
            # reset everything
            core.wait(0.15)
            logging.warning(f'START_CONF')
            logging.flush()
            event.clearEvents()
            mouse.clickReset()
            mouse.setVisible(False)
            probe_txt_stim.text = f"How confident are you that the target appears on the {choice.upper()} side most often?"
            txt_idx = 0
            trg_rgb = dp.DPxTriggerToRGB(600)
            while True: # --------------------------------------------------- CONF ONSET

                # keep track of quadrants
                quad_idx = (txt_idx % 4)

                # draw trigger stimulus
                if txt_idx < 4:
                    #trig_stim.lineColor = trg_rgb
                    #trig_stim.start = TLC_QUAD[quad_idx]
                    #trig_stim.end= [TLC_QUAD[quad_idx][0] + 1, TLC_QUAD[quad_idx][1]]
                    trig_stim.color = trg_rgb
                    trig_stim.pos = TLC_QUAD[quad_idx]                    
                    trig_stim.draw()

                # # set positions
                # probe_txt_stim.pos = CENTRE_HIGH_QUAD[quad_idx]
                # scale_conf.pos = CENTRE_QUAD[quad_idx]
                # certain_txt_stim.pos = SCALE_UP_LEFT_QUAD[quad_idx]
                # guess_txt_stim.pos = SCALE_DOWN_LEFT_QUAD[quad_idx]

                # # set marker location
                # mousePos = mouse.getPos()
                # mousePos_reformated = reformat_for_propixx(win, [mousePos[0], mousePos[1]])
                # if mousePos_reformated[quad_idx][1] >= SCALE_UP_LEFT_QUAD[quad_idx][1]:
                #     mousePos_reformated[quad_idx][1] = SCALE_UP_LEFT_QUAD[quad_idx][1]
                # if mousePos_reformated[quad_idx][1] <= SCALE_DOWN_LEFT_QUAD[quad_idx][1]:
                #     mousePos_reformated[quad_idx][1] = SCALE_DOWN_LEFT_QUAD[quad_idx][1]
                # mousePos_reformated[quad_idx][0] = CENTRE_QUAD[quad_idx][0]
                # marker_stim.pos = mousePos_reformated[quad_idx]

                # set positions
                probe_txt_stim.pos = CENTRE_HIGH_QUAD[quad_idx]
                scale_prob.pos = CENTRE_QUAD[quad_idx]
                guess_txt_stim.pos = SCALE_START_QUAD[quad_idx]
                certain_txt_stim.pos = SCALE_END_QUAD[quad_idx]

                # set marker location
                mousePos = mouse.getPos()
                mousePos_reformated = reformat_for_propixx(win, [mousePos[0], mousePos[1]])

                if mousePos_reformated[quad_idx][0] <= SCALE_START_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = SCALE_START_QUAD[quad_idx][0]
                if mousePos_reformated[quad_idx][0] >= SCALE_END_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = SCALE_END_QUAD[quad_idx][0]
                mousePos_reformated[quad_idx][1] = CENTRE_QUAD[quad_idx][1]
                marker_stim.pos = mousePos_reformated[quad_idx]

                # collect response
                buttons, times = mouse.getPressed(getTime=True)
                if buttons[0]: #if mouse gets pressed
                    low = guess_txt_stim.pos[0]
                    high = certain_txt_stim.pos[0]
                    rating = marker_stim.pos[0]
                    confidence = (rating + abs(low)) / (high + abs(low))
                    confidence_rt = times[0]
                    break

                # draw
                probe_txt_stim.draw()
                #scale_conf.draw()
                scale_prob.draw()
                certain_txt_stim.draw()
                guess_txt_stim.draw()
                marker_stim.draw()

                # flip window once fourth quadrant is drawn
                if quad_idx == 3: 
                    win.flip()

                # udpate frame
                txt_idx += 1 
    
            # update probe text
            probe_txt_stim.text = f"On what percentage of trials do you think the target appears on the {choice.upper()} side?"
            
            # reset everything
            logging.warning(f'START_PROB')
            logging.flush()
            event.clearEvents()
            mouse.clickReset()
            core.wait(0.15)  
            txt_idx = 0
            trg_rgb = dp.DPxTriggerToRGB(700)
            while True: # --------------------------------------------------- PROB ONSET

                # keep track of quadrants
                quad_idx = (txt_idx % 4)

                # draw trigger stimulus
                if txt_idx < 4:
                    #trig_stim.lineColor = trg_rgb
                    #trig_stim.start = TLC_QUAD[quad_idx]
                    #trig_stim.end= [TLC_QUAD[quad_idx][0] + 1, TLC_QUAD[quad_idx][1]]
                    trig_stim.color = trg_rgb
                    trig_stim.pos = TLC_QUAD[quad_idx]                    
                    trig_stim.draw()

                # set positions
                probe_txt_stim.pos = CENTRE_HIGH_QUAD[quad_idx]
                scale_prob.pos = CENTRE_QUAD[quad_idx]
                fifty_txt_stim.pos = SCALE_START_QUAD[quad_idx]
                hundred_txt_stim.pos = SCALE_END_QUAD[quad_idx]

                # set marker location
                mousePos = mouse.getPos()
                mousePos_reformated = reformat_for_propixx(win, [mousePos[0], mousePos[1]])

                if mousePos_reformated[quad_idx][0] <= SCALE_START_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = SCALE_START_QUAD[quad_idx][0]
                if mousePos_reformated[quad_idx][0] >= SCALE_END_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = SCALE_END_QUAD[quad_idx][0]
                mousePos_reformated[quad_idx][1] = CENTRE_QUAD[quad_idx][1]
                marker_stim.pos = mousePos_reformated[quad_idx]

                # collect response
                buttons, times = mouse.getPressed(getTime=True)
                if buttons[0]: #if mouse gets pressed
                    zero = fifty_txt_stim.pos[0]
                    one = hundred_txt_stim.pos[0]
                    rating = marker_stim.pos[0]
                    probability = (rating + abs(zero)) / (one + abs(zero))
                    probability_rt = times[0]
                    break

                # draw
                probe_txt_stim.draw()
                scale_prob.draw()
                fifty_txt_stim.draw()
                hundred_txt_stim.draw()
                marker_stim.draw()
                
                # flip window once fourth quadrant is drawn
                if quad_idx == 3: 
                    win.flip()

                # udpate frame
                txt_idx += 1 

        # reset everything
        event.clearEvents()
        mouse.clickReset()
        mouse.setVisible(False)
        core.wait(0.5) # pause to let participants find response buttongs again   
        # send trial number trigger ----------------------------------------------------TRG ONSET
        trg_rgb = dp.DPxTriggerToRGB(trial + 1)
        for frame in range(0, 4): 

            if frame == 0:
                logging.warning('TRL_TRG')
                logging.flush()

            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw trigger stimulus
            #trig_stim.lineColor = trg_rgb
            #trig_stim.start = TLC_QUAD[quad_idx]
            #trig_stim.end= [TLC_QUAD[quad_idx][0] + 1, TLC_QUAD[quad_idx][1]]
            trig_stim.color = trg_rgb
            trig_stim.pos = TLC_QUAD[quad_idx]            
            trig_stim.draw()

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()
        
        ## set stimulus properties for this trial 
        
        # set target and distractor orientations
        # if exp_target_side[runningTrialNo] == 'left':
        #     grt_stim1.ori = exp_target_oris[runningTrialNo]
        #     grt_stim2.ori = exp_distractor_oris[runningTrialNo]
        # elif exp_target_side[runningTrialNo] == 'right':
        #     grt_stim2.ori = exp_target_oris[runningTrialNo]
        #     grt_stim1.ori = exp_distractor_oris[runningTrialNo]

        grt_stim1.ori = exp_trial_seq[runningTrialNo][0][0]
        grt_stim2.ori = exp_trial_seq[runningTrialNo][0][1]

        # set tag properties
        opacity_idx_60 = phase_offsets[runningTrialNo]
        opacity_idx_64 = 0
        if exp_trial_seq[runningTrialNo][1][0] == 60:
            freq1 = opacity_60_Hz
            freq1_idx = opacity_idx_60
            freq2 = opacity_64_Hz 
            freq2_idx = opacity_idx_64
        elif exp_trial_seq[runningTrialNo][1][0] == 64:
            freq1 = opacity_64_Hz 
            freq1_idx = opacity_idx_64
            freq2 = opacity_60_Hz
            freq2_idx = opacity_idx_60

        # create fresh mask texture each trial
        noiseTexture = np.random.rand(32, 32) * 2.0 - 1
        msk_stim1.tex=noiseTexture
        msk_stim2.tex=noiseTexture

        # set random ITI duration
        ITI_DUR = np.random.randint(ITI_RANGE[0], ITI_RANGE[1])
        # if (runningTrialNo + 1) % N_TRIALS_CONF == 1:
        #     ITI_DUR += int(0.5*REFRATE)*4 # add another 0.5s to ITI if the previous trial has a confidence probe
        
        # find starting trigger value for this trial
        trg_val = find_trigger_prefix(runningTrialNo, exp_target_ratio, exp_hpl, exp_target_side) 
        trg_rgb = dp.DPxTriggerToRGB(trg_val) # convert to RGB

        # housekeeping
        win.frameClock.reset()
        win.frameIntervals = []
        event.clearEvents()
        logging.warning(f'START_TRIAL_{trial}')
        logging.flush()
        # start stimulus presentation -------------------------------------------------- ITI ONSET
        for frame in range(0, ITI_DUR): 

            if frame == 0:
                logging.warning('START_ISI')
                logging.flush()

            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw trigger stimulus
            if frame < 4:
                #trig_stim.lineColor = trg_rgb
                #trig_stim.start = TLC_QUAD[quad_idx]
                #trig_stim.end= [TLC_QUAD[quad_idx][0] + 1, TLC_QUAD[quad_idx][1]]
                trig_stim.color = trg_rgb
                trig_stim.pos = TLC_QUAD[quad_idx]                
                trig_stim.draw()

            # draw fixation
            fix_stim.pos = CENTRE_QUAD[quad_idx]
            fix_stim.draw()

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()
        
        # update trigger value
        trg_val += 10
        trg_rgb = dp.DPxTriggerToRGB(trg_val)
        for frame in range(0, PLH_DUR): # ------------------------------------------------ PLACEHOLDER ONSET

            if frame == 0:
                logging.warning('START_PHD')
                logging.flush()
            
            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw trigger stimulus
            if frame < 4:
                #trig_stim.lineColor = trg_rgb
                #trig_stim.start = TLC_QUAD[quad_idx]
                #trig_stim.end= [TLC_QUAD[quad_idx][0] + 1, TLC_QUAD[quad_idx][1]]
                trig_stim.color = trg_rgb
                trig_stim.pos = TLC_QUAD[quad_idx]
                trig_stim.draw()

            # draw fixation
            fix_stim.pos = CENTRE_QUAD[quad_idx]
            fix_stim.draw()

            # set placeholder positions
            plh_stim1.pos = LEFT_QUAD[quad_idx]
            plh_stim2.pos = RIGHT_QUAD[quad_idx]

            # set placeholder opacity
            #plh_stim1.opacity = freq1[freq1_idx]
            #plh_stim2.opacity = freq2[freq2_idx]

            # draw placeholders
            plh_stim1.draw()
            plh_stim2.draw()

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()
            
            # update tags
            freq1_idx += 1
            freq2_idx += 1

        # reset things before target display
        event.clearEvents()
        clock.reset()
        # update trigger value
        trg_val += 10
        trg_rgb = dp.DPxTriggerToRGB(trg_val)
        for frame in range(0, TAR_DUR): # ------------------------------------------------ TARGET ONSET

            if frame == 0:
                logging.warning('START_TAR')
                logging.flush()

            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw trigger stimulus
            if frame < 4:
                #trig_stim.lineColor = trg_rgb
                #trig_stim.start = TLC_QUAD[quad_idx]
                #trig_stim.end= [TLC_QUAD[quad_idx][0] + 1, TLC_QUAD[quad_idx][1]]
                trig_stim.color = trg_rgb
                trig_stim.pos = TLC_QUAD[quad_idx]
                trig_stim.draw()

            # draw fixation
            fix_stim.pos = CENTRE_QUAD[quad_idx]
            fix_stim.draw()

            # set placeholder positions
            plh_stim1.pos = LEFT_QUAD[quad_idx]
            plh_stim2.pos = RIGHT_QUAD[quad_idx]
            
            # set placeholder opacity
            #plh_stim1.opacity = freq1[freq1_idx]
            #plh_stim2.opacity = freq2[freq2_idx]
            
            # set grating positions
            grt_stim1.pos = LEFT_QUAD[quad_idx]
            grt_stim2.pos = RIGHT_QUAD[quad_idx]

            # set grating opacity
            #grt_stim1.opacity = freq1[freq1_idx]
            #grt_stim2.opacity = freq2[freq2_idx]

            # draw placeholders
            plh_stim1.draw()
            plh_stim2.draw()

            # draw gratings
            grt_stim1.draw()
            grt_stim2.draw()

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

            # update tags
            freq1_idx += 1
            freq2_idx += 1

        # update trigger value
        trg_val += 10
        trg_rgb = dp.DPxTriggerToRGB(trg_val)
        for frame in range(0, MSK_DUR): # ------------------------------------------------ MASK ONSET

            if frame == 0:
                logging.warning('START_MSK')
                logging.flush()

            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw trigger stimulus
            if frame < 4:
                #trig_stim.lineColor = trg_rgb
                #trig_stim.start = TLC_QUAD[quad_idx]
                #trig_stim.end= [TLC_QUAD[quad_idx][0] + 1, TLC_QUAD[quad_idx][1]]
                trig_stim.color = trg_rgb
                trig_stim.pos = TLC_QUAD[quad_idx]
                trig_stim.draw()
            
            # draw fixation
            fix_stim.pos = CENTRE_QUAD[quad_idx]
            fix_stim.draw()

            # set placeholder positions 
            plh_stim1.pos = LEFT_QUAD[quad_idx]
            plh_stim2.pos = RIGHT_QUAD[quad_idx]

            # set mask positions
            msk_stim1.pos = LEFT_QUAD[quad_idx]
            msk_stim2.pos = RIGHT_QUAD[quad_idx]
            
            # set placeholder opacity
            #plh_stim1.opacity = freq1[freq1_idx]
            #plh_stim2.opacity = freq2[freq2_idx]

            # set mask opacity
            #msk_stim1.opacity = freq1[freq1_idx]
            #msk_stim2.opacity = freq2[freq2_idx]

            # draw placeholders
            plh_stim1.draw()
            plh_stim2.draw()

            # draw masks
            msk_stim1.draw()
            msk_stim2.draw()

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

            # update tags
            freq1_idx += 1
            freq2_idx += 1

        for frame in range(0, RSP_DUR): # ------------------------------------------------ RESPONSE ONSET

            if frame == 0:
                logging.warning('START_RSP')
                logging.flush()
            
            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw fixation
            fix_stim.pos = CENTRE_QUAD[quad_idx]
            fix_stim.draw()

            # set placeholder positions and fill
            plh_stim1.pos = LEFT_QUAD[quad_idx]
            plh_stim2.pos = RIGHT_QUAD[quad_idx]

            # set placeholder opacity
            #plh_stim1.opacity = freq1[freq1_idx]
            #plh_stim2.opacity = freq2[freq2_idx]

            # draw placeholders
            plh_stim1.draw()
            plh_stim2.draw()

            # collect response input
            quitPressed = event.getKeys(keyList = QUITKEYS)
            pressed = event.getKeys(keyList = RESPKEYS, timeStamped = clock)
            if quitPressed: # exit task 
                dp.DPxSetPPxDlpSeqPgrm('RGB')
                dp.DPxDisableDoutPixelMode()
                dp.DPxWriteRegCache()
                dp.DPxClose() 
                core.quit()
            elif pressed:
                logging.warning('RESPONSE')
                logging.flush()
                response = pressed[0][0]
                rt = pressed[0][1]
                break

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

            # update tags
            freq1_idx += 1
            freq2_idx += 1
            
        # calculate correct response
        tar_idx = np.isin(exp_trial_seq[runningTrialNo][0], [45, -45])
        target_ori = exp_trial_seq[runningTrialNo][0][tar_idx]
        if target_ori == -45:
            correct_response = 'c'
        elif target_ori == 45:
            correct_response = 'm'
        # calculate accuracy and points
        if pressed:
            if response == correct_response:
                    acc = 1
                    points = POINTS_COR
            elif response != correct_response:
                    acc = 0
                    points = 0
            # update feedback text
            if acc == 1:
                txt_stim.text = 'Correct! 10 points.'
                # update trigger value
                trg_val += 10
            elif acc == 0:
                txt_stim.text = 'Incorrect! No points.'
                # update trigger value
                trg_val += 20
        elif not pressed:
            response = 999
            rt = 999
            acc = 999
            points = 0
            txt_stim.text = 'Too slow! No points.'
            # update trigger value
            trg_val += 30
        
        trg_rgb = dp.DPxTriggerToRGB(trg_val)
        for frame in range(0, FDB_DUR): # --------------------------------------------------- FDB ONSET

            if frame == 0:
                logging.warning('START_FDB')
                logging.flush()

            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw trigger stimulus
            if frame < 4:
                #trig_stim.lineColor = trg_rgb
                #trig_stim.start = TLC_QUAD[quad_idx]
                #trig_stim.end= [TLC_QUAD[quad_idx][0] + 1, TLC_QUAD[quad_idx][1]]
                trig_stim.color = trg_rgb
                trig_stim.pos = TLC_QUAD[quad_idx]
                trig_stim.draw()

            # draw
            txt_stim.pos = CENTRE_QUAD[quad_idx]
            txt_stim.draw()
                
            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

        # calculate reward
        cents_this_trial = points*CENTS_PER_POINT
        points_this_block += points
        cents_this_block += cents_this_trial
        cumulative_cents += cents_this_trial
        cumulative_points += points

        # update experiment handler data to save
        exp.addData('Trial', trial + 1)
        exp.addData('RunningTrialNo', runningTrialNo + 1)
        exp.addData('Block', block)
        exp.addData('TargetRatio', exp_target_ratio[runningTrialNo])
        exp.addData('HPL', exp_hpl[runningTrialNo])
        exp.addData('TargetSide', exp_target_side[runningTrialNo])
        exp.addData('TargetOri', target_ori[0])
        exp.addData('StimLeftOri', exp_trial_seq[runningTrialNo][0][0])
        exp.addData('StimRightOri', exp_trial_seq[runningTrialNo][0][1])
        exp.addData('StimLeftFreq', exp_trial_seq[runningTrialNo][1][0])
        exp.addData('StimRightFreq', exp_trial_seq[runningTrialNo][1][1])
        exp.addData('60HzPhaseOffset', phase_offsets[runningTrialNo])
        exp.addData('Response', response)
        exp.addData('RT', np.round(rt, 3))
        exp.addData('Accuracy', acc)
        exp.addData('Points', points)
        exp.addData('Choice', choice)
        exp.addData('ChoiceRT', np.round(choice_rt, 3))
        exp.addData('Confidence', np.round(confidence, 3))  
        exp.addData('ConfidenceRT', np.round(confidence_rt, 3))  
        exp.addData('Probability', np.round(probability, 3))  
        exp.addData('ProbabilityRT', np.round(probability_rt, 3))    
        exp.addData('CumulativePoints', cumulative_points)
        exp.addData('Cents', cents_this_trial)
        exp.addData('CumulativeCents', cumulative_cents)
        exp.nextEntry() # move to next line in data output
        runningTrialNo += 1
        framesPerTrial[f'Trial_{trial + 1}'] = win.frameIntervals

    # take a break and save data at the end of the block --------------------------------------------------
    win.recordFrameIntervals = False
    exp.saveAsWideText(
        fileName = FILEPATH, 
        appendFile=None,
        fileCollisionMethod='overwrite')
    framesPerBlock[f'Block_{block}'] = framesPerTrial
    with open(FRMSFILEPATH, 'w') as file:
        file.write(json.dumps(framesPerBlock)) 

    # present end of block text
    if block < N_BLOCKS:
        txt_stim.text = f'''
        End of Block {block}. Take a break. \n\n 
        You got {points_this_block} points in that block! Great job! \n\n
        Press space when you're ready to continue.''' 
    elif block == N_BLOCKS:
        txt_stim.text = f'''
        End of Block {block}. You're all done! \n\n 
        You got {cumulative_points} points! \n\n
        Press space and contact the experimenter.''' 
    event.clearEvents()
    txt_idx = 0
    while True:

        # keep track of quadrants
        quad_idx = (txt_idx % 4)

        # draw
        txt_stim.pos = CENTRE_QUAD[quad_idx]
        txt_stim.draw()

        # collect user input to exit 
        pressed = event.getKeys(keyList = CONKEYS)
        if pressed:
            break

        # flip window once fourth quadrant is drawn
        if quad_idx == 3: 
            win.flip()

        # udpate frame
        txt_idx += 1 
    
## ======================================================================
## End experiment
## ======================================================================
logging.warning('END_EXPERIMENT')
logging.flush()
event.clearEvents()
dp.DPxSetPPxDlpSeqPgrm('RGB')
dp.DPxDisableDoutPixelMode()
dp.DPxWriteRegCache()
dp.DPxClose()
win.close()

## ======================================================================
## Run debriefing
## ======================================================================
if DEBRIEF: 
    debrief = {
        'I acknowledge that I have been appropriately debriefed and shown a copy of the debriefing questions': False,
        'Name': '',
        'Date': ''}

    if expInfo['Debrief']:
        dlg = gui.DlgFromDict(
            debrief, 
            title = 'Acknowledgement of Debriefing')
        if not dlg.OK:
            core.quit()

    # create experiment handler to save output
    dbrf = data.ExperimentHandler(name=EXPERIMENT,
                                extraInfo=debrief)
    dbrf.dataNames = []
    identifier = datetime.now().timestamp()
    identifier = str.split(str(identifier), '.')[0]    
    dbrf.saveAsWideText(
            fileName = DATAPATH + f'AcknowledgementOfDebriefing_{identifier}.txt', 
            appendFile=None,
            fileCollisionMethod='overwrite')

## ======================================================================
## Check for dropped frames
## ======================================================================
thold = 1/REFRATE + 0.002
thold = win.refreshThreshold
frms_all = []
with open(FRMSFILEPATH) as file:
    frms = json.loads(file.read())
for block in frms.keys():
    for trial in frms[block].keys():
        frms_all.append(frms[block][trial][:])
frms_all = np.concatenate(frms_all)
dropped_frames = sum(frms_all > thold)
dropped_frames_pcnt = np.round(dropped_frames/len(frms_all), 2)
plt.plot(frms_all, marker = 'o', ms = 0.5, ls = '')
plt.hlines(thold, xmin=0, xmax=len(frms_all), color = 'black', ls = '--')
plt.hlines(1/REFRATE, xmin=0, xmax=len(frms_all), color = 'black', )
plt.hlines((1/REFRATE)*2, xmin=0, xmax=len(frms_all), color = 'black')
plt.title(f'Dropped frames: {dropped_frames} of {len(frms_all)} ({dropped_frames_pcnt}%)')
plt.ylabel('Frame duration (s)')
plt.xlabel('Frame')
plt.show()
print(f'Dropped frames: {dropped_frames} of {len(frms_all)} ({dropped_frames_pcnt}%)')



