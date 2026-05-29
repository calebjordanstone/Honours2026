from datetime import datetime
from psychopy import core, event, visual, logging, gui, event, monitors, data 
from pathlib import Path
from propixx_functions import *
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import json
from pypixxlib import _libdpx as dp
logging.console.setLevel(logging.WARNING)

## ======================================================================
## Initialise PROPixx
## ======================================================================
# establish connection to hardware
dp.DPxOpen()
isReady = dp.DPxIsReady()
if isReady:
    dp.DPxSetPPxDlpSeqPgrm('RGB') # set to 4x refresh rate, will set this later
    dp.DPxEnableDoutPixelMode() # enable pixel mode for triggers
    dp.DPxEnablePPxRearProjection() # enable rear projection to reverse display
    dp.DPxWriteRegCache()
else:
    print('Warning! DPx call failed, check connection to hardware')
    core.quit()
## ======================================================================
## Get experiment and monitor information 
## ======================================================================
expInfo = {
    'Monitor Name': 'test',
    'Monitor Refresh Rate': '120',
    'Subject Number': '',
    'Screen Number': '1',
    'Full Screen': True,
    'Demo': True,
    'Demographics': True,
    'Debrief': True,
    'Instructions': True,
    'Eyetracking': True}

dlg = gui.DlgFromDict(
    expInfo, 
    title = 'Experiment Information')

if not dlg.OK:
    core.quit()

EXPERIMENT = 'Honours2026'
DATETIME = datetime.now().strftime("%y%m%d%H%M") 
SUBID = 'sub-' + f'{int(expInfo["Subject Number"])}'.zfill(2)
REFRATE = int(expInfo['Monitor Refresh Rate'])
SCREEN = int(expInfo['Screen Number'])
FULLSCREEN = expInfo['Full Screen']
DEMO = expInfo['Demo']
DEMOGRAPHICS = expInfo['Demographics']
DEBRIEF = expInfo['Debrief']
INSTRUCTIONS = expInfo['Instructions']
EYETRACKING = expInfo['Eyetracking']

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
        'How do you describe your sex?': ['Woman/Female', 'Man/Male', 'Prefer not to answer', 'Non-binary', 'Other'],
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
        print(f'Eligibility criteria not met. Please contact the experimenter.\n\nPHOTOSENSITIVE = {PHOTOSENSITIVE}\nVISION OKAY = {VISION}\nHEARING OKAY = {HEARING}\nAGE = {AGE}')
        core.quit()

## ======================================================================
## Create data paths
## ======================================================================
#DATAPATH = 'C:/Users/cstone/OneDrive - UNSW/Documents/Projects/honours_projects/Honours2026/data/'
DATAPATH = '/home/experimenter/Experiments/Honours2026/data/'
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
## Create window, stimuli, and other experiment objects
## ======================================================================
# create window
win = visual.Window(
        screen = SCREEN, 
        monitor = MONITOR,
        size = MONITOR.getSizePix(),
        fullscr=1,
        pos = [0, 0],
        color = [0, 0, 0], # black or [-1, -1, -1]
        units = 'pix', 
        colorSpace = 'rgb255', # TODO: CHECK IF THIS NEEDS TO BE 'rgb'
        blendMode = 'avg',
        waitBlanking=False)
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
x_pos = dva_to_pix(6, **size_params) # stmuli to appear 6 degrees to left/right of centre
LEFT = [-x_pos, -y_pos]
RIGHT = [x_pos, -y_pos]

# define positions for text stimuli
CENTRE = [0, 0]
CENTRE_HIGH = [0, y_pos]
CENTRE_REALLY_HIGH = [0, y_pos*1.5]

# define positions for left/right scale stimuli
scale_y = dva_to_pix(2, **size_params)
scale_length = dva_to_pix(12, **size_params)
NOTCH_1_TEXT = [-scale_length/2, -scale_y]
NOTCH_2_TEXT = [-scale_length/2 + scale_length*0.2, -scale_y]
NOTCH_3_TEXT = [-scale_length/2 + scale_length*0.4, -scale_y]
NOTCH_4_TEXT = [-scale_length/2 + scale_length*0.6, -scale_y]
NOTCH_5_TEXT = [-scale_length/2 + scale_length*0.8, -scale_y]
NOTCH_6_TEXT = [scale_length/2, -scale_y]
NOTCH_1 = [-scale_length/2, 0]
NOTCH_2 = [-scale_length/2 + scale_length*0.2, 0]
NOTCH_3 = [-scale_length/2 + scale_length*0.4, 0]
NOTCH_4 = [-scale_length/2 + scale_length*0.6, 0]
NOTCH_5 = [-scale_length/2 + scale_length*0.8, 0]
NOTCH_6 = [scale_length/2, 0]

# define position for trigger stimulus
TLC = [-win.size[0]/2 + 1, win.size[1]/2 - 1] # top left corner

# define positions for calibration
CALIB_POINT_1 = [-(win.size[0]/10)*2.5, (win.size[1]/10)*2.5]
CALIB_POINT_2 = [(win.size[0]/10)*2.5, (win.size[1]/10)*2.5]
CALIB_POINT_3 = [0, 0]
CALIB_POINT_4 = [-(win.size[0]/10)*2.5, -(win.size[1]/10)*2.5]
CALIB_POINT_5 = [(win.size[0]/10)*2.5, -(win.size[1]/10)*2.5]

# create new positions for PROPIXX
LEFT_QUAD = reformat_for_propixx(win, LEFT)
RIGHT_QUAD = reformat_for_propixx(win, RIGHT)
LEFT_RIGHT_QUAD = [j for i in zip(LEFT_QUAD, RIGHT_QUAD) for j in i]
CENTRE_QUAD = reformat_for_propixx(win, CENTRE) 
CENTRE_HIGH_QUAD = reformat_for_propixx(win, CENTRE_HIGH)
CENTRE_REALLY_HIGH_QUAD = reformat_for_propixx(win, CENTRE_REALLY_HIGH)
TLC_QUAD = reformat_for_propixx(win, TLC)
NOTCH_1_TEXT_QUAD = reformat_for_propixx(win, NOTCH_1_TEXT)
NOTCH_2_TEXT_QUAD = reformat_for_propixx(win, NOTCH_2_TEXT)
NOTCH_3_TEXT_QUAD = reformat_for_propixx(win, NOTCH_3_TEXT)
NOTCH_4_TEXT_QUAD = reformat_for_propixx(win, NOTCH_4_TEXT)
NOTCH_5_TEXT_QUAD = reformat_for_propixx(win, NOTCH_5_TEXT)
NOTCH_6_TEXT_QUAD = reformat_for_propixx(win, NOTCH_6_TEXT)
NOTCH_1_QUAD = reformat_for_propixx(win, NOTCH_1)
NOTCH_2_QUAD = reformat_for_propixx(win, NOTCH_2)
NOTCH_3_QUAD = reformat_for_propixx(win, NOTCH_3)
NOTCH_4_QUAD = reformat_for_propixx(win, NOTCH_4)
NOTCH_5_QUAD = reformat_for_propixx(win, NOTCH_5)
NOTCH_6_QUAD = reformat_for_propixx(win, NOTCH_6)
notchQuadList = [NOTCH_1_QUAD, NOTCH_2_QUAD, NOTCH_3_QUAD,
                 NOTCH_4_QUAD, NOTCH_5_QUAD, NOTCH_6_QUAD]
notchTextQuadList = [NOTCH_1_TEXT_QUAD, NOTCH_2_TEXT_QUAD, NOTCH_3_TEXT_QUAD,
                     NOTCH_4_TEXT_QUAD, NOTCH_5_TEXT_QUAD, NOTCH_6_TEXT_QUAD]
CALIB_POINT_1_QUAD = reformat_for_propixx(win, CALIB_POINT_1)
CALIB_POINT_2_QUAD = reformat_for_propixx(win, CALIB_POINT_2)
CALIB_POINT_3_QUAD = reformat_for_propixx(win, CALIB_POINT_3)
CALIB_POINT_4_QUAD = reformat_for_propixx(win, CALIB_POINT_4)
CALIB_POINT_5_QUAD = reformat_for_propixx(win, CALIB_POINT_5)
CALIB_POINTS = [CALIB_POINT_1_QUAD, 
                CALIB_POINT_2_QUAD,
                CALIB_POINT_3_QUAD,
                CALIB_POINT_4_QUAD,
                CALIB_POINT_5_QUAD]
# CALIB_POINTS_QUAD = [j for i in zip(CALIB_POINT_1_QUAD, 
#                                     CALIB_POINT_2_QUAD,
#                                     CALIB_POINT_3_QUAD,
#                                     CALIB_POINT_4_QUAD,
#                                     CALIB_POINT_5_QUAD) for j in i]
# define colours
ORANGE = [194, 99, 32] 
WHITE = [255, 255, 255]
GREY = [80, 80, 80]
DARK_GREY = [50, 50, 50]
BLACK = [0, 0, 0]

# define response keys
RESPKEYS = ['c', 'm']
QUITKEYS = ['escape']
CONKEYS = ['space']
EXITKEYS = ['return']

# define number of trials/blocks
N_PRACTICE_TRIALS = 20 
N_TRIALS = 64 
N_BLOCKS = 15 #20
N_STATES = 8
N_STATE_CHANGES = N_STATES - 1
N_TRIALS_PER_STATE = int((N_TRIALS * N_BLOCKS) / N_STATES)
N_TRIALS_CONF = 8 

probe_trials = []
for half_state in np.arange(N_STATES*2):
    x = [59, 1]
    while sum(x) >= 60:
        x = np.random.randint(5,15,7)
    y = np.zeros(60)
    idx = 0
    for _, j in enumerate(x):
        idx += j
        y[idx] = 1  
    probe_trials.extend(y)  


# define target ratios
HIGH = [0.9, 0.1]
MED_HIGH = [0.8, 0.2]
MED_LOW = [0.7, 0.3]
LOW = [0.6, 0.4]

# define points
POINTS_COR = 10

# define timing 
ITI_RANGE = [int(0.3*REFRATE), int(0.6*REFRATE)]
PLH_DUR = int(1.3*REFRATE) 
TAR_DUR = int(0.067*REFRATE)
MSK_DUR = int(0.067*REFRATE)
RSP_DUR = int(0.866*REFRATE)
FDB_DUR = int(0.3*REFRATE)*4 # multiply  by 4 to account for increase in refrate from PROPixx when presenting text stimulus

# create oscillation for stimuli
framerate = REFRATE * 4
duration = 3 # seconds
t = np.tile(np.linspace(0, 1, framerate, endpoint=False), duration) 
fs = [60, 64] #[framerate/8, framerate/7] # set frequencies 
a = 0.5 # ampltiude
p = 0 # phase
opacity_60_Hz = a*(np.sin(2*np.pi*fs[0]*t + p)) + 0.5
opacity_64_Hz = a*(np.sin(2*np.pi*fs[1]*t + p)) + 0.5
t_calib = np.tile(np.linspace(0, 0.5, framerate, endpoint=False), 300) 
calib_size = a*(np.sin(2*np.pi*0.75*t_calib + 90)) + 0.6

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
fix_array = visual.ElementArrayStim(
    win=win,
    opacities = 1,
    contrs = 1,
    nElements= 4,
    elementTex=None,
    elementMask='cross',
    xys=CENTRE_QUAD,
    units='pix',
    oris=0,
    colors=WHITE,
    colorSpace = 'rgb255',
    sizes=fix_stim_height/3) ## TODO: CHECK ACTUAL STIM HEIGHTS IN DVA
trig_stim = visual.Rect(
    height=1,
    width=1,
    #size=1,
    lineColor=None, 
    interpolate=False,
    **stim_params)
trg_array = visual.ElementArrayStim(
    win=win,
    units="pix",
    colorSpace='rgb255',
    nElements=4,
    opacities=1,
    contrs=1,
    oris=0,
    elementTex=None,
    elementMask=None,
    xys= TLC_QUAD,
    sizes=1)

# create target, distractor, and placeholder stims
# note: for the below stimuli, suffix "1" denotes left and suffix "2" denotes right side of the screen
crc_stim_rad = dva_to_pix(dva=3, **size_params) ## TODO: CHECK SIZING OF THESE STIMS AND THE IMAGE STIMS
plh_stim1 = visual.Circle(
    lineColor = WHITE,
    fillColor = WHITE,
    radius = crc_stim_rad,
    edges = 100,
    lineWidth = 1,
    **stim_params)
plh_stim2 = visual.Circle(
    lineColor = WHITE,
    fillColor = WHITE,
    radius = crc_stim_rad,
    edges = 100,
    lineWidth = 1,
    **stim_params)
plh_array = visual.ElementArrayStim(
    win=win,
    units="pix",
    colorSpace='rgb',
    nElements=8,
    opacities=1,
    contrs=1,
    elementTex='plh.png',
    elementMask=None,
    xys=LEFT_RIGHT_QUAD,
    sizes=crc_stim_rad)
grt_stim1 = visual.GratingStim(
    tex = 'sin',
    mask = 'gauss',
    maskParams = {"sd": 3.8},
    sf = .1,
    size = crc_stim_rad*2,
    ori = 45,
    **stim_params)
grt_stim2 = visual.GratingStim(
    tex = 'sin',
    mask = 'gauss',
    maskParams = {"sd": 3.8},
    sf = .1,
    size = crc_stim_rad*2,
    ori = 0,
    **stim_params)
grt_array = visual.ElementArrayStim(
    win=win,
    units="pix",
    colorSpace='rgb',
    nElements=8,
    opacities=1,
    contrs=1,
    oris=0,
    elementTex='grt.png',
    elementMask=None,
    xys=LEFT_RIGHT_QUAD,
    sizes=crc_stim_rad)
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
msk_array = visual.ElementArrayStim(
    win=win,
    units="pix",
    colorSpace='rgb',
    nElements=8,
    opacities=1,
    contrs=1,
    oris=0,
    elementTex='msk.png',
    elementMask=None,
    xys=LEFT_RIGHT_QUAD,
    sizes=crc_stim_rad)

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
    size=(scale_length, scale_length/20),
    **stim_params)
N_NOTCHES = 6
notchList = [
    visual.Rect(color=WHITE,
                size=(scale_length/20, scale_length/10),
                **stim_params)
    for notch in range(N_NOTCHES)]
# fifty_txt_stim = visual.TextStim(
#                             text = '50%', 
#                             color = WHITE,
#                             height = txt_stim_height, 
#                             **stim_params)
# sixty_txt_stim = visual.TextStim(
#                             text = '60%', 
#                             color = WHITE,
#                             height = txt_stim_height, 
#                             **stim_params)
# seventy_txt_stim = visual.TextStim(
#                             text = '70%', 
#                             color = WHITE,
#                             height = txt_stim_height, 
#                             **stim_params)
# eighty_txt_stim = visual.TextStim(
#                             text = '80%', 
#                             color = WHITE,
#                             height = txt_stim_height, 
#                             **stim_params)
# ninety_txt_stim = visual.TextStim(
#                             text = '90%', 
#                             color = WHITE,
#                             height = txt_stim_height, 
#                             **stim_params)
# hundred_txt_stim = visual.TextStim(
#                             text = '100%', 
#                             color = WHITE,
#                             height = txt_stim_height, 
#                             **stim_params)
always_txt_stim = visual.TextStim(
                            text = 'Always', 
                            color = WHITE,
                            height = txt_stim_height, 
                            **stim_params)
half_txt_stim = visual.TextStim(
                            text = 'Half the time', 
                            color = WHITE,
                            height = txt_stim_height, 
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
mkr_stim_rad = dva_to_pix(dva=0.75, **size_params)
marker_stim = visual.Circle(
    edges=100,
    radius=mkr_stim_rad,
    **stim_params)
marker_stim.color = ORANGE

# create stims for eye-tracker calibration
calib_array = visual.ElementArrayStim(
    win=win,
    units='pix',
    colorSpace = 'rgb255',
    colors=WHITE,
    nElements=4,
    elementTex=None,
    elementMask='circle',
    #xys=CALIB_POINTS_QUAD,
    sizes=crc_stim_rad/6)

# create list of all stimuli to use later ## Add trig stim?
stims = [txt_stim, 
         fix_stim,
         plh_stim1, plh_stim2,
         grt_stim1, grt_stim2, 
         msk_stim1, msk_stim2, 
         probe_txt_stim,
         left_txt_stim, right_txt_stim,
         scale_prob, marker_stim,
         guess_txt_stim, certain_txt_stim,
         always_txt_stim, half_txt_stim,
        trig_stim] 

# rescale stimuli to half size to account for resolution drop
for stim in stims:
    stim.size = stim.size/2 
for notch in notchList:
    notch.size = notch.size/2

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

# write instruction text
instructText = {

'INSTRUCTIONS_1': '''
    Welcome to our experiment! \n
    When the task begins, you will see two circles on the bottom half of the screen, as shown below. \n
    Press space to continue. 
    ''',

'INSTRUCTIONS_2': '''
    These circles will then rapidly display a target and a distractor. \n
    The TARGET will be TILTED 45 degrees, either to the left or the right. \n
    The distractor will be oriented either vertically or horizontally. \n
    Press space to continue.
    ''',

'INSTRUCTIONS_3': '''
    Your first goal is to respond to the TARGET'S ORIENTATION. \n
    Press "C" if the target is tilted LEFT. Press "M" if the target is tilted RIGHT. \n
    Note that you are responding based on whether the target is TILTED left or right, not if it is located on the left or right side of the screen. \n
    Press the correct key to continue. 
    ''',

'INSTRUCTIONS_4': '''
    Great, now for your second goal. \n
    Throughout the experiment, you may notice different "phases" in which the target appears on one side of the screen more often. We'll call this the TARGET'S PREFERRED LOCATION. \n
    Importantly, in the different phases of the experiment, the target's preferred location, and how often it appears there, will CHANGE. \n
    Periodically, you will be asked 3 questions about your understanding of what phase you are currently in. You will be asked these questions multiple times per phase, so you should use all the trials you think belong to the current phase to inform your answers. \n
    You will need to use the mouse to respond to these questions. \n
    Press space to see what the first question will be.
    ''',

'INSTRUCTIONS_5': '''
    The first question will be: \n
    "On which side do you think the target appears most often?" \n
    Have a go at selecting an answer below. 
    ''',

'INSTRUCTIONS_6': '''
    The second question will be: \n
    "How confident are you that the target appears on the {CHOICE} side most often?" \n
    Have a go at using the scale below.  
    ''',

'INSTRUCTIONS_7': '''
    The third question will be: \n
    "How often do you think the target appears on the {CHOICE} side?" \n
    Have a go at using the scale below.  
    ''',

'INSTRUCTIONS_8': '''
    Great, you're ready to go! \n\n
    Throughout the experiment, please keep your chin on the chinrest and your eyes focussed on the fixation cross. \n\n
    To begin, there will be a practice phase and then the experimentation phase will follow. \n\n
    Press space to continue.     
    '''
}
## ======================================================================
## Create trial structure
## ======================================================================
stim_tag_combos_tar_left = np.repeat(
    np.asarray(
        tuple(
            itertools.product(
                [-45, 45],
                [(60, 64), 
                 (64, 60)])),
            dtype = 'object'), 
    repeats=1,
    axis=0)

stim_tag_combos_tar_right = np.repeat(
    np.asarray(
        tuple(
            itertools.product(
                [-45, 45], 
                [(60, 64), 
                 (64, 60)])),
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
            [np.repeat(stim_tag_combos_tar_right, repeats=int(n_trials_high/4), axis=0), # repeats = number of trials needed, divided by the number of trial types with the target on the HPL side
             np.repeat(stim_tag_combos_tar_left, repeats=int(n_trials_low/4), axis=0)]
            )
    elif hpl == 'left':
        state_trial_seq = np.concatenate(
            [np.repeat(stim_tag_combos_tar_left, repeats=int(n_trials_high/4), axis=0), # repeats = number of trials needed, divided by the number of trial types with the target on the HPL side
             np.repeat(stim_tag_combos_tar_right, repeats=int(n_trials_low/4), axis=0)]
            )
    #np.random.shuffle(state_trial_seq)
    # find target side on each trial
    target_seq = np.concatenate([np.ones(int(n_trials_high)), np.zeros(int(n_trials_low))])
    #np.random.shuffle(target_seq)
    state_target_side = []
    for trial in target_seq:
        if hpl == 'right':
            if trial == 1:
                target_side = ['right']
            else:
                target_side = ['left']
        elif hpl == 'left':
            if trial == 1: 
                target_side = ['left']
            else:
                target_side = ['right']
        state_target_side.extend(target_side)
        #exp_target_side.extend(target_side)
    trial_plus_target = np.concatenate([np.array(state_trial_seq), 
                                        np.expand_dims(np.array(state_target_side), 1)], axis=1)
    np.random.shuffle(trial_plus_target)
    # add to whole experiment array
    exp_target_side.extend(list(trial_plus_target[:, 2]))
    exp_target_ratio.extend([ratio]*N_TRIALS_PER_STATE)
    exp_hpl.extend([hpl]*N_TRIALS_PER_STATE)
    exp_trial_seq.extend(trial_plus_target[:, 0:2])

# create random phase offsets
phase_offsets = np.repeat(np.arange(1, 9), repeats=(N_TRIALS*N_BLOCKS)/8) # offset phase by 1-8 samples from the cycle
np.random.shuffle(phase_offsets)

# create random sequence of distractors
exp_distractor_seq = np.repeat([0, 90], repeats=(N_TRIALS*N_BLOCKS)/2)
np.random.shuffle(exp_distractor_seq)

# create sequence of trial numbers for each state
state_trial_nos = np.tile(np.arange(1, N_TRIALS_PER_STATE + 1), reps = N_STATES)

# define trigger values
'''
Block number triggers:
230 + block number. e.g., 231 = block 1

Trial number triggers:
1 to N_TRIALS. e.g., 1 = trial 1

Trial structure triggers:
Ratio: H = high, MH = medium high, ML = medium low, L = low
High-probability location: L = left, R = right
Target location: l = left, r = right

ITI: x0 # inter-trial interval onset
PLH: x2 # placeholder onsest
TAR: x4 # target onset
MSK: x6 # mask onset
RES: x7 # missed response
RES: x8 # correct response
RES: x9 # incorrect response

HLl: 07x
HLr: 08x
HRl: 09x
HRr: 10x

MHLl: 11x
MHLr: 12x
MHRl: 13x
MHRr: 14x

MLLl: 15x
MLLr: 16x
MLRl: 17x
MLRr: 18x

LLl: 19x
LLr: 20x
LRl: 21x
LRr: 22x

Probe Questions:
choice: 251
confidence: 252
probability: 253

Save eyetracking data:
254
Exit eyetracker:
255

'''

def find_trigger_prefix(trl_idx, exp_target_ratio, exp_hpl, exp_target_side):

    # find prefix for state ratio for current trial
    if (exp_target_ratio[trl_idx] == HIGH) & (exp_hpl[trl_idx] == 'left') & (exp_target_side[trl_idx] == 'left'):
        trig = 70
    elif (exp_target_ratio[trl_idx] == HIGH) & (exp_hpl[trl_idx] == 'left') & (exp_target_side[trl_idx] == 'right'):
        trig = 80
    elif (exp_target_ratio[trl_idx] == HIGH) & (exp_hpl[trl_idx] == 'right') & (exp_target_side[trl_idx] == 'left'):
        trig = 90
    elif (exp_target_ratio[trl_idx] == HIGH) & (exp_hpl[trl_idx] == 'right') & (exp_target_side[trl_idx] == 'right'):
        trig = 100

    elif (exp_target_ratio[trl_idx] == MED_HIGH) & (exp_hpl[trl_idx] == 'left') & (exp_target_side[trl_idx] == 'left'):
        trig = 110
    elif (exp_target_ratio[trl_idx] == MED_HIGH) & (exp_hpl[trl_idx] == 'left') & (exp_target_side[trl_idx] == 'right'):
        trig = 120
    elif (exp_target_ratio[trl_idx] == MED_HIGH) & (exp_hpl[trl_idx] == 'right') & (exp_target_side[trl_idx] == 'left'):
        trig = 130
    elif (exp_target_ratio[trl_idx] == MED_HIGH) & (exp_hpl[trl_idx] == 'right') & (exp_target_side[trl_idx] == 'right'):
        trig = 140

    elif (exp_target_ratio[trl_idx] == MED_LOW) & (exp_hpl[trl_idx] == 'left') & (exp_target_side[trl_idx] == 'left'):
        trig = 150
    elif (exp_target_ratio[trl_idx] == MED_LOW) & (exp_hpl[trl_idx] == 'left') & (exp_target_side[trl_idx] == 'right'):
        trig = 160
    elif (exp_target_ratio[trl_idx] == MED_LOW) & (exp_hpl[trl_idx] == 'right') & (exp_target_side[trl_idx] == 'left'):
        trig = 170
    elif (exp_target_ratio[trl_idx] == MED_LOW) & (exp_hpl[trl_idx] == 'right') & (exp_target_side[trl_idx] == 'right'):
        trig = 180

    elif (exp_target_ratio[trl_idx] == LOW) & (exp_hpl[trl_idx] == 'left') & (exp_target_side[trl_idx] == 'left'):
        trig = 190
    elif (exp_target_ratio[trl_idx] == LOW) & (exp_hpl[trl_idx] == 'left') & (exp_target_side[trl_idx] == 'right'):
        trig = 200
    elif (exp_target_ratio[trl_idx] == LOW) & (exp_hpl[trl_idx] == 'right') & (exp_target_side[trl_idx] == 'left'):
        trig = 210
    elif (exp_target_ratio[trl_idx] == LOW) & (exp_hpl[trl_idx] == 'right') & (exp_target_side[trl_idx] == 'right'):
        trig = 220

    # return trigger
    return trig


    # if exp_target_ratio[trl_idx] == HIGH:
    #     pfx_1 = 1
    # elif exp_target_ratio[trl_idx] == MED_HIGH:
    #     pfx_1 = 2
    # elif exp_target_ratio[trl_idx] == MED_LOW:
    #     pfx_1 = 3
    # else: 
    #     pfx_1 = 4
    # # find prefix for HPL side for current trial
    # if exp_hpl[trl_idx] == 'left':
    #     pfx_2 = 1
    # else:
    #     pfx_2 = 2
    # # find prefix for target side for current trial
    # if exp_target_side[trl_idx] == 'left':
    #     pfx_3 = 1
    # else: 
    #     pfx_3 = 2
    # # add zeros to end
    # sfx = '00'
    # # combine into one string
    # trig = f"{pfx_1}{pfx_2}{pfx_3}{sfx}"
    # # return as integer
    # return int(trig)

## ======================================================================
## Present stimuli
## ======================================================================
dp.DPxSetPPxDlpSeqPgrm('QUAD4X') # set to 4x refresh rate
dp.DPxUpdateRegCache()

if INSTRUCTIONS:

    ## Present instructions 
    for j, txt in enumerate(instructText.keys()):

        mouse.clickReset()
        event.clearEvents()
        logging.warning(f'{txt}')
        txt_stim.text = instructText[txt]
        instr_idx = 0
        while True:

            # keep track of quadrants
            quad_idx = (instr_idx % 4)

            # present placeholder
            if j == 0:
                    
                # set positions
                txt_stim.pos = CENTRE_REALLY_HIGH_QUAD[quad_idx]
                fix_stim.pos = CENTRE_QUAD[quad_idx]
                plh_stim1.pos = LEFT_QUAD[quad_idx]
                plh_stim2.pos = RIGHT_QUAD[quad_idx]

                # draw placeholders
                txt_stim.draw()
                plh_stim1.draw()
                plh_stim2.draw()
                fix_stim.draw()

                # collect user input to exit 
                pressed = event.getKeys(keyList = CONKEYS)
                if pressed: 
                    break

            # present targets
            elif j == 1:
                    
                # set positions
                txt_stim.pos = CENTRE_REALLY_HIGH_QUAD[quad_idx]
                fix_stim.pos = CENTRE_QUAD[quad_idx]
                plh_stim1.pos = LEFT_QUAD[quad_idx]
                plh_stim2.pos = RIGHT_QUAD[quad_idx]
                grt_stim1.pos = LEFT_QUAD[quad_idx]
                grt_stim2.pos = RIGHT_QUAD[quad_idx]

                # draw placeholders
                txt_stim.draw()
                plh_stim1.draw()
                plh_stim2.draw()
                grt_stim1.draw()
                grt_stim2.draw()
                fix_stim.draw()

                # collect user input to exit 
                pressed = event.getKeys(keyList = CONKEYS)
                if pressed: 
                    break

            # respond to targets
            elif j == 2:
                    
                # set positions
                txt_stim.pos = CENTRE_REALLY_HIGH_QUAD[quad_idx]
                fix_stim.pos = CENTRE_QUAD[quad_idx]
                plh_stim1.pos = LEFT_QUAD[quad_idx]
                plh_stim2.pos = RIGHT_QUAD[quad_idx]
                grt_stim1.pos = LEFT_QUAD[quad_idx]
                grt_stim2.pos = RIGHT_QUAD[quad_idx]

                # draw placeholders
                txt_stim.draw()
                plh_stim1.draw()
                plh_stim2.draw()
                grt_stim1.draw()
                grt_stim2.draw()
                fix_stim.draw()

                # collect user input to exit 
                pressed = event.getKeys(keyList = 'm')
                if pressed: 
                    break
            
            # present probe instructions
            elif j == 3:

                txt_stim.pos = CENTRE_QUAD[quad_idx]
                txt_stim.draw()

                # collect user input to exit 
                pressed = event.getKeys(keyList = CONKEYS)
                if pressed: 
                    break
            
            # present first probe question
            elif j == 4:

                mouse.setVisible(True)

                # set positions
                txt_stim.pos = CENTRE_REALLY_HIGH_QUAD[quad_idx]
                left_txt_stim.pos = LEFT_QUAD[quad_idx]
                right_txt_stim.pos = RIGHT_QUAD[quad_idx]

                # draw
                txt_stim.draw()
                left_txt_stim.draw()
                right_txt_stim.draw()

                # collect response input
                buttons, times = mouse.getPressed(getTime=True)
                if buttons[0]: #if mouse gets pressed
                    mousePos = mouse.getPos() # get mouse position
                    if left_txt_stim.contains(mousePos):
                        choice = "left"
                        mouse.setVisible(False)
                        instructText['INSTRUCTIONS_6'] = instructText['INSTRUCTIONS_6'].format(CHOICE = choice.upper())
                        instructText['INSTRUCTIONS_7'] = instructText['INSTRUCTIONS_7'].format(CHOICE = choice.upper())
                        core.wait(0.15)
                        break
                    elif right_txt_stim.contains(mousePos):
                        choice = "right"
                        mouse.setVisible(False)
                        instructText['INSTRUCTIONS_6'] = instructText['INSTRUCTIONS_6'].format(CHOICE = choice.upper())
                        instructText['INSTRUCTIONS_7'] = instructText['INSTRUCTIONS_7'].format(CHOICE = choice.upper())
                        core.wait(0.15)
                        break

        
            # present second probe question
            elif j == 5:
                        
                # set positions
                txt_stim.pos = CENTRE_REALLY_HIGH_QUAD[quad_idx]
                scale_prob.pos = CENTRE_QUAD[quad_idx]
                guess_txt_stim.pos = NOTCH_1_TEXT_QUAD[quad_idx]
                certain_txt_stim.pos = NOTCH_6_TEXT_QUAD[quad_idx]
                for i, notch in enumerate(notchList):
                    notch.pos = notchQuadList[i][quad_idx]

                # set marker location
                mousePos = mouse.getPos()
                mousePos_reformated = reformat_for_propixx(win, [mousePos[0], mousePos[1]])

                if mousePos_reformated[quad_idx][0] <= NOTCH_1_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = NOTCH_1_QUAD[quad_idx][0]
                if mousePos_reformated[quad_idx][0] >= NOTCH_6_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = NOTCH_6_QUAD[quad_idx][0]
                mousePos_reformated[quad_idx][1] = CENTRE_QUAD[quad_idx][1]
                marker_stim.pos = mousePos_reformated[quad_idx]

                # collect response
                buttons, times = mouse.getPressed(getTime=True)
                if buttons[0]: #if mouse gets pressed
                    core.wait(0.15)
                    break

                # draw
                txt_stim.draw()
                scale_prob.draw()
                for notch in notchList:
                    notch.draw()
                certain_txt_stim.draw()
                guess_txt_stim.draw()
                marker_stim.draw()

            # present third probe question
            elif j == 6:

                # set positions
                txt_stim.pos = CENTRE_REALLY_HIGH_QUAD[quad_idx]
                scale_prob.pos = CENTRE_QUAD[quad_idx]
                half_txt_stim.pos = NOTCH_1_TEXT_QUAD[quad_idx]
                always_txt_stim.pos = NOTCH_6_TEXT_QUAD[quad_idx]
                for i, notch in enumerate(notchList):
                    notch.pos = notchQuadList[i][quad_idx]

                # set marker location
                mousePos = mouse.getPos()
                mousePos_reformated = reformat_for_propixx(win, [mousePos[0], mousePos[1]])

                if mousePos_reformated[quad_idx][0] <= NOTCH_1_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = NOTCH_1_QUAD[quad_idx][0]
                if mousePos_reformated[quad_idx][0] >= NOTCH_6_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = NOTCH_6_QUAD[quad_idx][0]
                mousePos_reformated[quad_idx][1] = CENTRE_QUAD[quad_idx][1]
                marker_stim.pos = mousePos_reformated[quad_idx]

                # collect response
                buttons, times = mouse.getPressed(getTime=True)
                if buttons[0]: #if mouse gets pressed
                    core.wait(0.15)
                    break

                # draw
                txt_stim.draw()
                scale_prob.draw()
                for notch in notchList:
                    notch.draw()
                half_txt_stim.draw()
                always_txt_stim.draw()
                marker_stim.draw()
            
            # present final instruction
            else:

                # set positions
                txt_stim.pos = CENTRE_QUAD[quad_idx]

                # draw text
                txt_stim.draw()

                # collect response input
                pressed = event.getKeys(keyList = CONKEYS)
                if pressed: 
                    break

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

            # udpate frame
            instr_idx += 1 

event.clearEvents()
logging.flush()
## Start practice trials 
if DEMO:

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
                    guess_txt_stim.pos = NOTCH_1_TEXT_QUAD[quad_idx]
                    certain_txt_stim.pos = NOTCH_6_TEXT_QUAD[quad_idx]
                    for i, notch in enumerate(notchList):
                        notch.pos = notchQuadList[i][quad_idx]

                    # set marker location
                    mousePos = mouse.getPos()
                    mousePos_reformated = reformat_for_propixx(win, [mousePos[0], mousePos[1]])

                    if mousePos_reformated[quad_idx][0] <= NOTCH_1_QUAD[quad_idx][0]:
                        mousePos_reformated[quad_idx][0] = NOTCH_1_QUAD[quad_idx][0]
                    if mousePos_reformated[quad_idx][0] >= NOTCH_6_QUAD[quad_idx][0]:
                        mousePos_reformated[quad_idx][0] = NOTCH_6_QUAD[quad_idx][0]
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
                    for notch in notchList:
                        notch.draw()
                    certain_txt_stim.draw()
                    guess_txt_stim.draw()
                    marker_stim.draw()

                    # flip window once fourth quadrant is drawn
                    if quad_idx == 3: 
                        win.flip()

                    # udpate frame
                    txt_idx += 1 
        
                # update probe text
                probe_txt_stim.text = f"How often do you think the target appears on the {choice.upper()} side?"
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
                    half_txt_stim.pos = NOTCH_1_TEXT_QUAD[quad_idx]
                    always_txt_stim.pos = NOTCH_6_TEXT_QUAD[quad_idx]
                    for i, notch in enumerate(notchList):
                        notch.pos = notchQuadList[i][quad_idx]

                    # set marker location
                    mousePos = mouse.getPos()
                    mousePos_reformated = reformat_for_propixx(win, [mousePos[0], mousePos[1]])

                    if mousePos_reformated[quad_idx][0] <= NOTCH_1_QUAD[quad_idx][0]:
                        mousePos_reformated[quad_idx][0] = NOTCH_1_QUAD[quad_idx][0]
                    if mousePos_reformated[quad_idx][0] >= NOTCH_6_QUAD[quad_idx][0]:
                        mousePos_reformated[quad_idx][0] = NOTCH_6_QUAD[quad_idx][0]
                    mousePos_reformated[quad_idx][1] = CENTRE_QUAD[quad_idx][1]
                    marker_stim.pos = mousePos_reformated[quad_idx]

                    # collect response
                    buttons, times = mouse.getPressed(getTime=True)
                    if buttons[0]: #if mouse gets pressed
                        zero = half_txt_stim.pos[0]
                        one = always_txt_stim.pos[0]
                        rating = marker_stim.pos[0]
                        probability = (rating + abs(zero)) / (one + abs(zero))
                        probability_rt = times[0]
                        break

                    # draw
                    probe_txt_stim.draw()
                    scale_prob.draw()
                    for notch in notchList:
                        notch.draw()
                    half_txt_stim.draw()
                    always_txt_stim.draw()
                    marker_stim.draw()
                    
                    # flip window once fourth quadrant is drawn
                    if quad_idx == 3: 
                        win.flip()

                    # udpate frame
                    txt_idx += 1 

                # reset everything
                win.flip()
                mouse.clickReset()
                mouse.setVisible(False)
                event.clearEvents()
                core.wait(1.5) # pause to let participants find response buttons again   
        
            ## Set trial properites
            # set target and distractor orientations
            stim_options = [[-45, 0], [-45, 90], [45, 0], [45, 90], 
                            [0, -45], [0, 45], [90, -45], [90, 45]]
            stim_idx = np.random.choice(8)
            stim_oris = np.tile(stim_options[stim_idx], 4)
            grt_array.oris = stim_oris

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
            opacities = [j for i in zip(freq1, freq2) for j in i]

            # set random ITI duration
            ITI_DUR = np.random.randint(ITI_RANGE[0], ITI_RANGE[1])

            event.clearEvents()
            logging.warning('START_ISI')
            # start stimulus presentation -------------------------------------------------- ITI ONSET
            for frame in range(0, ITI_DUR): 

                # draw stimuli
                fix_array.draw()

                # flip sindow
                win.flip()

            logging.warning('START_PHD')
            for frame in range(0, PLH_DUR): # ------------------------------------------------ PLACEHOLDER ONSET

                # set opacities
                plh_array.opacities = opacities[tag_idx:tag_idx+8]

                # draw stimuli 
                fix_array.draw()
                plh_array.draw()

                # flip window
                win.flip()

                # update tag
                tag_idx += 8

            # reset things before target display
            event.clearEvents()
            clock.reset()
            logging.warning('START_TAR')
            for frame in range(0, TAR_DUR): # ------------------------------------------------ TARGET ONSET

                # set opacities
                grt_array.opacities = opacities[tag_idx:tag_idx+8]

                # draw stimuli 
                fix_array.draw()
                grt_array.draw()

                # flip window
                win.flip()

                # update tag
                tag_idx += 8

            logging.warning('START_MSK')
            for frame in range(0, MSK_DUR): # ------------------------------------------------ MASK ONSET

                # set opacities
                msk_array.opacities = opacities[tag_idx:tag_idx+8]

                # draw stimuli 
                fix_array.draw()
                msk_array.draw()

                # flip window
                win.flip()

                # update tag
                tag_idx += 8

            logging.warning('START_RSP')
            for frame in range(0, RSP_DUR): # ------------------------------------------------ RESPONSE ONSET
                
                # set opacities
                plh_array.opacities = opacities[tag_idx:tag_idx+8]

                # draw stimuli 
                fix_array.draw()
                plh_array.draw()

                # flip window
                win.flip()

                # update tag
                tag_idx += 8

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
                    response = pressed[0][0]
                    rt = pressed[0][1]
                    break

            # calculate correct response
            tar_idx = np.isin(stim_options[stim_idx], [45, -45])
            target_ori = np.array(stim_options[stim_idx])[tar_idx]
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

            logging.warning('START_FDB')
            for frame in range(0, FDB_DUR): # --------------------------------------------------- FDB ONSET

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
        End of practice block.\n\n
        Please wait for the experimenter.''' 
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
            if (continue_practice) or (end_practice):
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
txt_stim.text = 'You will now start the main experiment.\n\nPlease wait for the experimenter to start the EEG recording.'
instr_idx = 0
while True:

    # keep track of quadrants
    quad_idx = (instr_idx % 4)

    # draw
    txt_stim.pos = CENTRE_QUAD[quad_idx]
    txt_stim.draw()

    # flip window once fourth quadrant is drawn
    if quad_idx == 3: 
        win.flip()

    # collect user input to exit 
    pressed = event.getKeys(keyList = EXITKEYS)
    if pressed: 
        break

    # udpate frame
    instr_idx += 1 

if EYETRACKING:

    event.clearEvents()
    txt_stim.text = 'Start calibration'
    instr_idx = 0
    while True:

        # keep track of quadrants
        quad_idx = (instr_idx % 4)

        # draw
        txt_stim.pos = CENTRE_QUAD[quad_idx]
        txt_stim.draw()

        # flip window once fourth quadrant is drawn
        if quad_idx == 3: 
            win.flip()

        # collect user input to exit 
        begin_calibration = event.getKeys(keyList = CONKEYS)
        if begin_calibration:
            break

        # udpate frame
        instr_idx += 1
    
    # enter calibration
    while True:

        logging.warning('START_CALIBRATION')
        logging.flush()

        # Cycle through calibration points 
        for point, _ in enumerate(CALIB_POINTS):

            # set positions for current point
            calib_array.xys = CALIB_POINTS[point]

            # present on screen
            event.clearEvents()
            calib_idx = 0 
            while True:

                # draw stims 
                calib_array.sizes = (crc_stim_rad/6)*calib_size[calib_idx:calib_idx+4]
                calib_array.draw()

                # flip window
                win.flip()

                #collect user input to exit 
                pressed = event.getKeys(keyList = CONKEYS)
                if pressed:
                    break

                # update idx
                calib_idx += 4
        
        event.clearEvents()
        txt_stim.text = 'Repeat calibration?'
        instr_idx = 0
        while True:

            # keep track of quadrants
            quad_idx = (instr_idx % 4)

            # draw
            txt_stim.pos = CENTRE_QUAD[quad_idx]
            txt_stim.draw()

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

            # collect user input to exit 
            continue_calibration = event.getKeys(keyList = CONKEYS)
            end_calibration = event.getKeys(keyList = EXITKEYS)
            if (continue_calibration) or (end_calibration):
                break

            # udpate frame
            instr_idx += 1 

        if end_calibration:
            break

    event.clearEvents()
    txt_stim.text = 'Start validation'
    instr_idx = 0
    while True:

        # keep track of quadrants
        quad_idx = (instr_idx % 4)

        # draw
        txt_stim.pos = CENTRE_QUAD[quad_idx]
        txt_stim.draw()

        # flip window once fourth quadrant is drawn
        if quad_idx == 3: 
            win.flip()

        # collect user input to exit 
        begin_validation = event.getKeys(keyList = CONKEYS)
        if begin_validation:
            break

        # udpate frame
        instr_idx += 1

    # enter validation
    while True: 

        logging.warning('START_VALIDATION')
        logging.flush()

        # Cycle through calibration points 
        for point, _ in enumerate(CALIB_POINTS):

            # set positions for current point
            calib_array.xys = CALIB_POINTS[point]

            # present on screen
            event.clearEvents()
            calib_idx = 0 
            while True:

                # draw stims 
                calib_array.sizes = (crc_stim_rad/6)*calib_size[calib_idx:calib_idx+4]
                calib_array.draw()

                # flip window
                win.flip()

                #collect user input to exit 
                pressed = event.getKeys(keyList = CONKEYS)
                if pressed:
                    break

                # update idx
                calib_idx += 4
        
        event.clearEvents()
        txt_stim.text = 'Repeat validation?'
        instr_idx = 0
        while True:

            # keep track of quadrants
            quad_idx = (instr_idx % 4)

            # draw
            txt_stim.pos = CENTRE_QUAD[quad_idx]
            txt_stim.draw()

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()

            # collect user input to exit 
            continue_validation = event.getKeys(keyList = CONKEYS)
            end_validation = event.getKeys(keyList = EXITKEYS)
            if (continue_validation) or (end_validation):
                break

            # udpate frame
            instr_idx += 1 

        if end_validation:
            break

## Start experiment 
# set placeholder values for metacognitive questions
choice = np.nan 
confidence = np.nan
probability = np.nan
choice_rt = np.nan
confidence_rt = np.nan
probability_rt = np.nan
framesPerBlock = {}
runningTrialNo = 0
cumulative_points = 0
cumulative_cents = 0
logging.warning('START_EXP')
logging.flush()
for block in range(1, N_BLOCKS + 1): 

    # present instructions 
    txt_idx = 0
    txt_stim.text = f'This is Block {block}. \n\n Please keep your eyes on the fixation cross throughout each trial. \n\n Press space to begin.' 
    trg_rgb = dp.DPxTriggerToRGB(230 + block) # convert to RGB
    trg_array.colours = trg_rgb
    while True:

        # keep track of quadrants
        quad_idx = (txt_idx % 4)

        # draw trigger stimulus
        if txt_idx < 4:
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
        
        # update txt idx 
        txt_idx += 1

    # logging 
    logging.warning(f'START_BLOCK_{block}')
    logging.flush()
    win.recordFrameIntervals = True
    framesPerTrial = {}
    points_this_block = 0
    cents_this_block = 0
    # start presenting trials ---------------------------------------------------------- TRIAL ONSET
    for trial in range(0, N_TRIALS):
 
        # send trial number trigger ----------------------------------------------------TRG ONSET
        trg_rgb = dp.DPxTriggerToRGB(trial + 1)
        trg_array.colors = trg_rgb
        logging.warning('TRL_TRG')
        for frame in range(0, 1): 

            # draw stims 
            trg_array.draw()

            # flip window
            win.flip()
        
        ## set stimulus properties for this trial 
        if exp_target_side[runningTrialNo] == 'left':
            grt_stim1_ori = exp_trial_seq[runningTrialNo][0]
            grt_stim2_ori = exp_distractor_seq[runningTrialNo]
        elif exp_target_side[runningTrialNo] == 'right':
            grt_stim1_ori = exp_distractor_seq[runningTrialNo]
            grt_stim2_ori = exp_trial_seq[runningTrialNo][0]
        grt_array.oris = np.tile([grt_stim1_ori, grt_stim2_ori], 4)
        
        # set tag properties
        tag_idx = 0
        opacity_idx_60 = phase_offsets[runningTrialNo]
        opacity_idx_64 = 0
        if exp_trial_seq[runningTrialNo][1][0] == 60:
            freq1 = opacity_60_Hz
            freq1_idx = opacity_idx_60
            freq2 = opacity_64_Hz 
            freq2_idx = opacity_idx_64
            opacities = [j for i in zip(freq1[freq1_idx:], freq2[0:-freq1_idx]) for j in i] # sample from each frequency one at a time, so I can assign them sequentially to the left/right stims. Also trim the sine waves to be same length, taking from front of 60Hz because of it's phase offset, and back of 64Hz so length matches
        elif exp_trial_seq[runningTrialNo][1][0] == 64:
            freq1 = opacity_64_Hz 
            freq1_idx = opacity_idx_64
            freq2 = opacity_60_Hz
            freq2_idx = opacity_idx_60
            opacities = [j for i in zip(freq1[0:-freq2_idx], freq2[freq2_idx:]) for j in i]

        # set random ITI duration
        ITI_DUR = np.random.randint(ITI_RANGE[0], ITI_RANGE[1])

        # find starting trigger value for this trial
        trg_val = find_trigger_prefix(runningTrialNo, exp_target_ratio, exp_hpl, exp_target_side) 
        trg_rgb = dp.DPxTriggerToRGB(trg_val) # convert to RGB
        trg_array.colors = trg_rgb

        # housekeeping
        win.frameIntervals = []
        event.clearEvents()
        logging.warning(f'START_TRIAL_{trial}')
        logging.warning('START_ISI')
        win.frameClock.reset()
        # check processing times
        # t0 = core.getTime()
        # proc_times_iti = []
        # start stimulus presentation -------------------------------------------------- ITI ONSET
        for frame in range(0, ITI_DUR): 

            # draw stims
            if frame == 0:
                trg_array.draw()
            fix_array.draw()
            
            # check processing times
            # proc_times_iti += [core.getTime() - t0]

            # flip window
            win.flip()

            # check timing
            # t0 = core.getTime() 

        # check timing
        # t0 = core.getTime() 
        # proc_times_plh = []
        # update trigger value
        trg_val += 2
        trg_rgb = dp.DPxTriggerToRGB(trg_val)
        trg_array.colors = trg_rgb
        logging.warning('START_PLH')
        for frame in range(0, PLH_DUR): # ------------------------------------------------ PLACEHOLDER ONSET

            # set opacities
            plh_array.opacities = opacities[tag_idx:tag_idx+8]

            # draw stimuli 
            if frame == 0:
                trg_array.draw()
            fix_array.draw()
            plh_array.draw()

            # check processing times
            # proc_times_plh += [core.getTime() - t0]

            # flip window
            win.flip()

            # update tag
            tag_idx += 8
            # t0 = core.getTime() 

        # reset things before target display
        event.clearEvents()
        clock.reset()

        # update trigger value
        trg_val += 2
        trg_rgb = dp.DPxTriggerToRGB(trg_val)
        trg_array.colors = trg_rgb
        logging.warning('START_TAR')

        # check timing
        # t0 = core.getTime() 
        # proc_times_tar = []
        for frame in range(0, TAR_DUR): # ------------------------------------------------ TARGET ONSET

            # set opacities
            grt_array.opacities = opacities[tag_idx:tag_idx+8]

            # draw stimuli
            if frame == 0:
                trg_array.draw()
            fix_array.draw()
            grt_array.draw()

            # check processing times
            # proc_times_tar += [core.getTime() - t0]

            # flip window
            win.flip()

            # update tag
            tag_idx += 8
            # t0 = core.getTime()             

        # update trigger value
        trg_val += 2
        trg_rgb = dp.DPxTriggerToRGB(trg_val)
        trg_array.colors = trg_rgb
        logging.warning('START_MSK')
        # check timing
        # t0 = core.getTime() 
        # proc_times_msk = []
        for frame in range(0, MSK_DUR): # ------------------------------------------------ MASK ONSET

            # set opacities
            msk_array.opacities = opacities[tag_idx:tag_idx+8]

            # draw stimuli 
            if frame == 0:
                trg_array.draw()
            fix_array.draw()
            msk_array.draw()

            # check processing times
            # proc_times_msk += [core.getTime() - t0]

            # flip window
            win.flip()

            # update tag
            tag_idx += 8
            # t0 = core.getTime()             


        logging.warning('START_RSP')
        # check timing
        # t0 = core.getTime() 
        # proc_times_rsp = []
        for frame in range(0, RSP_DUR): # ------------------------------------------------ RESPONSE ONSET

            # set opacities
            plh_array.opacities = opacities[tag_idx:tag_idx+8]

            # draw stimuli 
            fix_array.draw()
            plh_array.draw()

            # check processing times
            # proc_times_rsp += [core.getTime() - t0]

            # flip window
            win.flip()

            # update tag
            tag_idx += 8
            # t0 = core.getTime()   

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
                response = pressed[0][0]
                rt = pressed[0][1]
                break
            
        # calculate correct response
        target_ori = exp_trial_seq[runningTrialNo][0]
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
                trg_val += 2
            elif acc == 0:
                txt_stim.text = 'Incorrect! No points.'
                # update trigger value
                trg_val += 3
        elif not pressed:
            response = 999
            rt = 999
            acc = 999
            points = 0
            txt_stim.text = 'Too slow! No points.'
            # update trigger value
            trg_val += 1

        trg_rgb = dp.DPxTriggerToRGB(trg_val)
        trig_stim.color = trg_rgb
        trg_array.colors = trg_rgb
        logging.warning('START_FDB')
        # t0 = core.getTime() 
        # proc_times_fdb = []
        for frame in range(0, FDB_DUR): # --------------------------------------------------- FDB ONSET

            # keep track of quadrants
            quad_idx = (frame % 4)

            # draw trigger stimulus
            # if frame < 4:
            #     trig_stim.pos = TLC_QUAD[quad_idx]
            #     trig_stim.draw()
            if frame == 0:
                trg_array.draw()

            # draw
            txt_stim.pos = CENTRE_QUAD[quad_idx]
            txt_stim.draw()

            # check timing
            # proc_times_fdb += [core.getTime() - t0]

            # flip window once fourth quadrant is drawn
            if quad_idx == 3: 
                win.flip()
                # t0 = core.getTime()   
        
        # present confidence rating screens after ever N_TRIALS_CONF trials
        #if (runningTrialNo + 1) % N_TRIALS_CONF == 0:
        if probe_trials[runningTrialNo] == 1:

            # mark probe trials
            # probe_trial = 1

            logging.warning(f'START_CHOICE')
            # logging.flush()
            mouse.setVisible(True)
            mouse.clickReset()
            event.clearEvents()
            probe_txt_stim.text = "On which side do you think the target appears most often?"
            txt_idx = 0
            trg_rgb = dp.DPxTriggerToRGB(251)
            trig_stim.color = trg_rgb
            while True: # --------------------------------------------------- LEFT/RIGHT ONSET

                # keep track of quadrants
                quad_idx = (txt_idx % 4)

                # draw trigger stimulus
                if txt_idx < 4:
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
            # logging.flush()
            event.clearEvents()
            mouse.clickReset()
            mouse.setVisible(False)
            probe_txt_stim.text = f"How confident are you that the target appears on the {choice.upper()} side most often?"
            txt_idx = 0
            trg_rgb = dp.DPxTriggerToRGB(252)
            trig_stim.color = trg_rgb
            while True: # --------------------------------------------------- CONF ONSET

                # keep track of quadrants
                quad_idx = (txt_idx % 4)

                # draw trigger stimulus
                if txt_idx < 4:
                    trig_stim.pos = TLC_QUAD[quad_idx]                    
                    trig_stim.draw()

                # set positions
                probe_txt_stim.pos = CENTRE_HIGH_QUAD[quad_idx]
                scale_prob.pos = CENTRE_QUAD[quad_idx]
                guess_txt_stim.pos = NOTCH_1_TEXT_QUAD[quad_idx]
                certain_txt_stim.pos = NOTCH_6_TEXT_QUAD[quad_idx]
                for i, notch in enumerate(notchList):
                    notch.pos = notchQuadList[i][quad_idx]

                # set marker location
                mousePos = mouse.getPos()
                mousePos_reformated = reformat_for_propixx(win, [mousePos[0], mousePos[1]])

                if mousePos_reformated[quad_idx][0] <= NOTCH_1_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = NOTCH_1_QUAD[quad_idx][0]
                if mousePos_reformated[quad_idx][0] >= NOTCH_6_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = NOTCH_6_QUAD[quad_idx][0]
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
                for notch in notchList:
                    notch.draw()
                certain_txt_stim.draw()
                guess_txt_stim.draw()
                marker_stim.draw()

                # flip window once fourth quadrant is drawn
                if quad_idx == 3: 
                    win.flip()

                # udpate frame
                txt_idx += 1 
    
            # update probe text
            probe_txt_stim.text = f"How often do you think the target appears on the {choice.upper()} side?"
            
            # reset everything
            logging.warning(f'START_PROB')
            #logging.flush()
            event.clearEvents()
            mouse.clickReset()
            core.wait(0.15)  
            txt_idx = 0
            trg_rgb = dp.DPxTriggerToRGB(253)
            trig_stim.color = trg_rgb
            while True: # --------------------------------------------------- PROB ONSET

                # keep track of quadrants
                quad_idx = (txt_idx % 4)

                # draw trigger stimulus
                if txt_idx < 4:
                    trig_stim.pos = TLC_QUAD[quad_idx]                    
                    trig_stim.draw()

                # set positions
                probe_txt_stim.pos = CENTRE_HIGH_QUAD[quad_idx]
                scale_prob.pos = CENTRE_QUAD[quad_idx]
                half_txt_stim.pos = NOTCH_1_TEXT_QUAD[quad_idx]
                always_txt_stim.pos = NOTCH_6_TEXT_QUAD[quad_idx]
                for i, notch in enumerate(notchList):
                    notch.pos = notchQuadList[i][quad_idx]

                # set marker location
                mousePos = mouse.getPos()
                mousePos_reformated = reformat_for_propixx(win, [mousePos[0], mousePos[1]])

                if mousePos_reformated[quad_idx][0] <= NOTCH_1_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = NOTCH_1_QUAD[quad_idx][0]
                if mousePos_reformated[quad_idx][0] >= NOTCH_6_QUAD[quad_idx][0]:
                    mousePos_reformated[quad_idx][0] = NOTCH_6_QUAD[quad_idx][0]
                mousePos_reformated[quad_idx][1] = CENTRE_QUAD[quad_idx][1]
                marker_stim.pos = mousePos_reformated[quad_idx]

                # collect response
                buttons, times = mouse.getPressed(getTime=True)
                if buttons[0]: #if mouse gets pressed
                    zero = half_txt_stim.pos[0]
                    one = always_txt_stim.pos[0]
                    rating = marker_stim.pos[0]
                    probability = (rating + abs(zero)) / (one + abs(zero))
                    probability_rt = times[0]
                    break

                # draw
                probe_txt_stim.draw()
                scale_prob.draw()
                for notch in notchList:
                    notch.draw()
                half_txt_stim.draw()
                always_txt_stim.draw()
                marker_stim.draw()
                
                # flip window once fourth quadrant is drawn
                if quad_idx == 3: 
                    win.flip()

                # udpate frame
                txt_idx += 1 

            # reset everything
            win.flip()
            mouse.clickReset()
            mouse.setVisible(False)
            event.clearEvents()
            core.wait(0.75) # pause to let participants find response buttons again  
        
        # else:
        #     probe_trial = 0

        # flush log
        logging.flush()

        # calculate reward
        points_this_block += points
        cumulative_points += points

        # update experiment handler data to save
        exp.addData('Trial', trial + 1)
        exp.addData('RunningTrialNo', runningTrialNo + 1)
        exp.addData('StateTrialNo', state_trial_nos[runningTrialNo])
        #exp.addData('ProbeTrial', probe_trial)
        exp.addData('ProbeTrial', probe_trials[runningTrialNo])
        exp.addData('Block', block)
        exp.addData('TargetRatio', exp_target_ratio[runningTrialNo])
        exp.addData('HPL', exp_hpl[runningTrialNo])
        exp.addData('TargetSide', exp_target_side[runningTrialNo])
        exp.addData('TargetOri', target_ori)
        exp.addData('DistractorOri', exp_distractor_seq[runningTrialNo])
        # exp.addData('StimLeftOri', exp_trial_seq[runningTrialNo][0][0])
        # exp.addData('StimRightOri', exp_trial_seq[runningTrialNo][0][1])
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
        exp.nextEntry() # move to next line in data output
        runningTrialNo += 1
        framesPerTrial[f'Trial_{trial + 1}'] = win.frameIntervals

        # check timings
        # print(f"Proc times iti: {proc_times_iti}")
        # print(f"Proc times plh: {proc_times_plh}")
        # print(f"Proc times tar: {proc_times_tar}")
        # print(f"Proc times msk: {proc_times_msk}")
        # print(f"Proc times rsp: {proc_times_rsp}")
        # print(f"Proc times fdb: {proc_times_fdb}")

    # take a break and save data at the end of the block --------------------------------------------------
    win.recordFrameIntervals = False
    exp.saveAsWideText(
        fileName = FILEPATH, 
        appendFile=None,
        fileCollisionMethod='overwrite')
    framesPerBlock[f'Block_{block}'] = framesPerTrial
    with open(FRMSFILEPATH, 'w') as file:
        file.write(json.dumps(framesPerBlock)) 
    
    # send save-eyetracker-data trigger 
    trg_rgb = dp.DPxTriggerToRGB(254)
    trg_array.colors = trg_rgb
    logging.warning('SAVE_EYE_TRG')
    for frame in range(0, 1): 

        # draw stims 
        trg_array.draw()

        # flip window
        win.flip()
    
    core.wait(1)
    # present end of block text
    if block < N_BLOCKS:
        txt_stim.text = f'''
        Block {block} of 15 complete. Take a break. \n\n 
        You got {points_this_block} points in that block! Great job! \n\n
        Press space when you're ready to continue.''' 
    elif block == N_BLOCKS:
        txt_stim.text = f'''
        End of Block {block}. You're all done! \n\n 
        You got {cumulative_points} points! \n\n
        Please contact the experimenter.''' 
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
# send save-eyetracker-data-and-exit trigger 
trg_rgb = dp.DPxTriggerToRGB(255)
trg_array.colors = trg_rgb
logging.warning('EXIT_EYE_TRG')
for frame in range(0, 1): 

    # draw stims 
    trg_array.draw()

    # flip window
    win.flip()

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
    dbrf.nextEntry() 
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