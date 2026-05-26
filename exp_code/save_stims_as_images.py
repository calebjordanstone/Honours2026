
from psychopy import core, event, visual, logging, gui, event, monitors, data 
from pathlib import Path
import numpy as np
import os
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
 

## ======================================================================
## Create window, stimuli, and other experiment features
## ======================================================================
# define stimuli positions
size_params = { # common parameters used to calculate stimulus size in pixels
    'distance': MONITOR.getDistance(), 
    'screen_res': MONITOR.getSizePix(), 
    'screen_width': MONITOR.getWidth()}
crc_stim_rad = dva_to_pix(dva=3, **size_params) #3.78 ,makes an image that is 256*256 pixels

# create window
win = visual.Window(
        screen = 1, 
        monitor = MONITOR,
        size = [crc_stim_rad*2, crc_stim_rad*2], #MONITOR.getSizePix(), #
        fullscr=0,
        pos = [0, 0],
        color =[0, 0, 0], # black [-1, -1, -1], #
        units = 'pix', 
        colorSpace = 'rgb255', #rgb255
        blendMode = 'avg',
        waitBlanking=False)

# set mouse to invisible
mouse = event.Mouse(visible=False)

# define colours
ORANGE = [194, 99, 32] 
WHITE = [255, 255, 255]
GREY = [80, 80, 80]
DARK_GREY = [50, 50, 50]
BLACK = [0, 0, 0]

# create stimuli for each trial
stim_params = { # common parameters used across stimuli
    'win': win, 
    'units': 'pix', 
    'opacity': 1,
    'contrast': 1,
    'colorSpace': 'rgb255'} #rgb255

# trig_stim = visual.Rect(
#     width=1, 
#     height=1,
#     lineColor=None, 
#     interpolate=False,
#     **stim_params)

# create the stimuli to use during experiment
plh_stim1 = visual.Circle(
    lineColor = [255, 255, 255], #[1, 1, 1], #WHITE,
    fillColor = WHITE, #[50, 50, 50], #[-0.5, -0.5, -0.5], #DARK_GREY,
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
    ori = 0,
    **stim_params)

noiseTexture = np.random.rand(32, 32) * 2.0 - 1
msk_stim1 = visual.GratingStim(
    mask="gauss",
    tex=noiseTexture,
    maskParams = {"sd": 3.8},
    size=crc_stim_rad*2, 
    interpolate=False, 
    autoLog=False, 
    **stim_params)


# Create your stimulus
# Draw your stimulus to the window and flip it to the front
plh_stim1.draw()
# grt_stim1.draw()
#msk_stim1.draw()
win.flip()
# Screenshot the window to get it as a PIL image
img = win.screenshot
# Convert it to a numpy array and flip upside down (texture y coordinates are inverse to window y coordinates)
tex = (np.flipud(np.array(img)) / 128 - 1)
#print(tex.min(), tex.max())
# Close the texture window so it doesn't interfere with your experiment
win.close()
img.save("plh.png")
 
# create window
win = visual.Window(
        screen = 1, 
        monitor = MONITOR,
        size = MONITOR.getSizePix(), 
        fullscr=0,
        pos = [0, 0],
        color = [0, 0, 0], # black [-1, -1, -1], #
        units = 'pix', 
        colorSpace = 'rgb255', #'rgb255
        blendMode = 'avg',
        waitBlanking=False)
win.refreshThreshold = 1/120 + 0.002

array = visual.ElementArrayStim(
    win=win,
    opacities = 1,
    contrs = 1,
    nElements= 2,
    elementTex='msk.png',
    elementMask=None,
    xys=[(200, 0), (-200, 200)],
    units='pix',
    oris=0,
    colorSpace = 'rgb',
    sizes=crc_stim_rad
)

array.draw()
win.flip()