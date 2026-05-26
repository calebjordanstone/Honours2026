from datetime import datetime
from psychopy import core, event, visual, logging, gui, event, monitors, data 
from pathlib import Path
from pypixxlib import _libdpx as dp
from propixx_functions import *

## ======================================================================
## Get experiment and monitor information 
## ======================================================================
# configure monitor
MONITOR = monitors.Monitor('test')

## ======================================================================
## Create data paths
## ======================================================================
# DATAPATH = 'C:/Users/cstone/OneDrive - UNSW/Documents/Projects/honours_projects/Honours2026/data/'
DATAPATH = '/home/experimenter/Experiments/Honours2026'
LOGFILENAME = 'frames_check.txt'
LOGFILEPATH = DATAPATH + LOGFILENAME
logging.LogFile(LOGFILEPATH)
logging.console.setLevel(logging.WARNING)
## ======================================================================
## Initialise PROPixx
## ======================================================================
# establish connection to hardware
dp.DPxOpen()
isReady = dp.DPxIsReady()
if isReady:
    dp.DPxSetPPxDlpSeqPgrm('QUAD4X') # set to 4x refresh rate, will set this later
    dp.DPxEnableDoutPixelMode() # enable pixel mode for triggers
    dp.DPxEnablePPxRearProjection() # enable rear projection to reverse display
    dp.DPxWriteRegCache()
else:
    print('Warning! DPx call failed, check connection to hardware')

## ======================================================================
## Create window, stimuli, and other experiment features
## ======================================================================
# define stimuli positions
size_params = { # common parameters used to calculate stimulus size in pixels
    'distance': MONITOR.getDistance(), 
    'screen_res': MONITOR.getSizePix(), 
    'screen_width': MONITOR.getWidth()}

# create window
win = visual.Window(
        screen = 1, 
        monitor = MONITOR,
        size = MONITOR.getSizePix(), 
        fullscr=1,
        pos = [0, 0],
        color =[0, 0, 0], # black [-1, -1, -1], #
        units = 'pix', 
        colorSpace = 'rgb255', #rgb255
        blendMode = 'avg',
        waitBlanking=False)

# set mouse to invisible
mouse = event.Mouse(visible=False)

# define colours
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]

# define position for trigger stimulus
TLC = [-win.size[0]/2 + 1, win.size[1]/2 -1] # top left corner
BLC = [-win.size[0]/2 + 25, -win.size[1]/2 +25] 
BLC_QUAD = reformat_for_propixx(win, BLC)
TLC_QUAD = reformat_for_propixx(win, TLC)

# create stimuli for each trial
stim_params = { # common parameters used across stimuli
    'win': win, 
    'units': 'pix', 
    'opacity': 1,
    'contrast': 1,
    'colorSpace': 'rgb255'} #rgb255

trg_array = visual.ElementArrayStim(
    win=win,
    units="pix",
    colorSpace='rgb255',
    nElements=4,
    opacities=1,
    colors = [255, 255, 255],
    contrs=1,
    oris=0,
    elementTex=None,
    elementMask=None,
    xys= TLC_QUAD,
    sizes=1)

rect_array = visual.ElementArrayStim(
    win=win,
    units="pix",
    colorSpace='rgb255',
    nElements=4,
    opacities=1,
    contrs=1,
    oris=0,
    elementTex=None,
    elementMask=None,
    xys=BLC_QUAD,
    sizes=25)



#trg_vals = np.arange(100, 100000, 100)
trg_vals = np.arange(0, 255, 1)
rect_cols = [WHITE, BLACK]*500
# check processing times
t0 = core.getTime()
proc_times = []
win.recordFrameIntervals = True
for i, _ in enumerate(trg_vals):

    # update trigger values
    trg_rgb = dp.DPxTriggerToRGB(trg_vals[i]) # convert to RGB
    logging.warning(f'TRIG VALUE: {trg_vals[i]}')
    trg_array.colors = trg_rgb
    
    # update rectangle value
    rect_array.colors = rect_cols[i]

    # draw stims
    rect_array.draw()
    trg_array.draw()

    # collect user input to exit 
    pressed = event.getKeys(keyList = ['space'])
    if pressed:
        break 
    
    # check processing times
    proc_times += [core.getTime() - t0]

    # flip window
    win.flip()

    # reset time
    t0 = core.getTime() 

# get frames
frames = win.frameIntervals
logging.warning(f'Frame intervals: {frames}')
logging.warning(f'Processing times: {proc_times}')
# flush log
logging.flush()
# shut everything down
win.close()
dp.DPxSetPPxDlpSeqPgrm('RGB')
dp.DPxDisableDoutPixelMode()
dp.DPxWriteRegCache()
dp.DPxClose() 
core.quit()