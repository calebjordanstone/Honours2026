import polars as pl
import pandas as pd
import mne
import re
import numpy as np 
import scipy as sp 
import sklearn as sk
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import cmath
import h5py as h5
%matplotlib qt

# load data
DERIVATIVES_DATA_PATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/honours_projects/Honours2026/data/derivatives/"
SOURCEDATA_DATA_PATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/honours_projects/Honours2026/data/sourcedata/"
srcDataEEG = sorted(Path(DERIVATIVES_DATA_PATH).glob('**/*.fif'))


# Create function to compute coherece, take from https://osf.io/yrh6v/files/gz65u
def compute_coherence(data, fqs):
       
       n_trials, n_chans, n_tpoints = data._data.shape
       freqs = fqs #[60, 64] #np.arange(56, 69)
       n_freqs = len(freqs)

       Cxy = np.empty([n_freqs, n_chans, n_tpoints])
       for f, freq in enumerate(freqs): 
              
              # filter data 
              iir_params = dict(order=4, ftype='butter')
              X = data.copy().filter(
                     l_freq=freq - 1.9, 
                     h_freq=freq + 1.9,
                     picks='eeg',
                     method='iir', 
                     iir_params=iir_params, 
                     phase='zero', 
                     fir_window='hamming')
              X = X.get_data()
              # create reference sinusoid
              sfreq = int(data.info['sfreq'])
              duration = 4 # seconds
              t = np.tile(np.linspace(0, 1, sfreq, endpoint=False), duration) 
              Y = np.sin(2*np.pi*freq*t + 0.5*np.pi)
              Y = Y[0:n_tpoints]
 
              # Hilbert
              Xh = sp.signal.hilbert(X)
              Yh = sp.signal.hilbert(Y)

              # magnitude
              Xm = np.abs(Xh)
              Ym = np.abs(Yh)

              # phase
              Xp = np.angle(Xh)
              Yp = np.angle(Yh)

              # phase difference
              delta_p = Xp - Yp 

              # iterate through channels
              for c in range(n_chans):
                     # iterate through time points
                     for t in range(n_tpoints):
                            # run coherence calulation
                            coh_fct = (np.abs(np.sum(Xm[:, c, t]*Ym[t]*cmath.e**(1j*delta_p[:, c, t]))/n_trials)**2) / (np.sum((Xm[:, c, t]**2)*(Ym[t]**2))/n_trials)
                            Cxy[f, c, t] = coh_fct
                     
       return Cxy

for path in srcDataEEG:

    ## get subID   
    subID = path.stem.split('_')[0]
    ## load data 
    epochs = mne.read_epochs(path) 
    beh_path = sorted(Path(DERIVATIVES_DATA_PATH).glob(f'**/Honours2026_allBeh_Choice2.csv'))
    beh = pd.read_csv(beh_path[0], sep=',')
    beh = beh[beh['Subject'] == subID]

    ## exclude behavioural data trials based on missing eeg triggers
    if subID == 'sub-03': # missing first trial
         beh = beh[1:]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)
    if subID == 'sub-04': # first 2 trials missing
         beh = beh[2:]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)
    if subID == 'sub-10': # 3rd trial of block 9 missing plh trigger
         beh = beh[~((beh['Block']==9) & (beh['Trial'] == 3))]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)
    if subID == 'sub-13': # 1st trial of block 9 missing plh trigger
         beh = beh[~((beh['Block']==9) & (beh['Trial'] == 1))]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)
    if subID == 'sub-16': # 1st trial of block 1 and 2nd trial of block 9 missing plh trigger
         beh = beh[~(((beh['Block']==1) & (beh['Trial'] == 1)) | ((beh['Block']==9) & (beh['Trial'] == 2)))]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)
    if subID == 'sub-18': # 1st trial of block 1 and 3rd trial of block 9 missing plh trigger
         beh = beh[~(((beh['Block']==1) & (beh['Trial'] == 1)) | ((beh['Block']==9) & (beh['Trial'] == 3)))]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)
    if subID == 'sub-21': # 1st trial of blocks 1 & 9
         beh = beh[~(((beh['Block']==1) & (beh['Trial'] == 1)) | ((beh['Block']==9) & (beh['Trial'] == 1)))]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)
    if subID == 'sub-32': # 1st trial of block 9
         beh = beh[~((beh['Block']==9) & (beh['Trial'] == 1))]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)
    if subID == 'sub-36': # trials 18, 19, & 20 of block 14 are missing
         beh = beh[~((beh['Block']==14) & (beh['Trial'].isin([18, 19, 20])))]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)
    if subID == 'sub-37': # trial 1 of block 1 and 3rd trial of block 9 missing
         beh = beh[~(((beh['Block']==1) & (beh['Trial'] == 1)) | ((beh['Block']==9) & (beh['Trial'] == 3)))]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)
    if subID == 'sub-38': # trial 1 of block 9 missing
         beh = beh[~((beh['Block']==9) & (beh['Trial'] == 1))]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)
    if subID == 'sub-40': # trial 1 and 2 of block 1 missing
         beh = beh[~((beh['Block']==1) & (beh['Trial'].isin([1, 2])))]
         beh['RunningTrialNo'] = np.arange(1, len(beh)+1)   
    ## re-set running trial number to account for python indexing starting at 0  
    beh['RunningTrialNo2'] = beh['RunningTrialNo'] - 1 

    ## create new varaibles
    #beh['Ratio'] = np.where(beh['TargetRatio'].isin(['[0.9, 0.1]', '[0.8, 0.2]']), 'high', 'low')
    conditions = [(beh['TargetRatio'] == '[0.6, 0.4]'),
                  (beh['TargetRatio'] == '[0.7, 0.3]'),
                  (beh['TargetRatio'] == '[0.8, 0.2]'),
                  (beh['TargetRatio'] == '[0.9, 0.1]')]
    choices = ["low", "medLow", "medHigh", "high"]
    beh['State'] = np.select(conditions, choices, default="nan")
    beh['Experience'] = np.where(beh['StateTrialNo'] <= 60, 'low', 'high')
    beh['ConfidenceTfmd'] = beh['Confidence']*0.5 + 0.5
    beh['ProbabilityTfmd'] = beh['Probability']*0.5 + 0.5
    beh['TargetHPLCongruency'] = np.where(beh['TargetSide'] == beh['HPL'], 'Congruent', 'Incongruent')
    conditions = [(beh['TargetSide'] == 'right') & (beh['Choice2'] == 'right'),
                  (beh['TargetSide'] == 'right') & (beh['Choice2'] == 'left'),
                  (beh['TargetSide'] == 'left') & (beh['Choice2'] == 'right'),
                  (beh['TargetSide'] == 'left') & (beh['Choice2'] == 'left')]
    choices = ["Congruent", "Incongruent", "Incongruent", "Congruent"]
    beh["TargetChoiceCongruency"] = np.select(conditions, choices, default="nan")
    conditions = [(beh['Choice2'] == 'right') & (beh['HPL'] == 'right'),
                  (beh['Choice2'] == 'right') & (beh['HPL'] == 'left'),
                  (beh['Choice2'] == 'left') & (beh['HPL'] == 'right'),
                  (beh['Choice2'] == 'left') & (beh['HPL'] == 'left')]
    choices = ["Congruent", "Incongruent", "Incongruent", "Congruent"]
    beh["ChoiceHPLCongruent"] = np.select(conditions, choices, default="nan")
    beh.loc[~np.isnan(beh['Confidence']), 'ConfidenceBin'] = np.where(
          sp.stats.zscore(
                beh.loc[~np.isnan(beh['Confidence']), 'ConfidenceTfmd'], 
                    nan_policy='omit') >= 0, 'high', 'low')
#     beh.loc[~np.isnan(beh['Confidence']),'medianConf'] = beh.groupby(
#          'TargetRatio')['Confidence'].transform('median')
#     beh['ConfidenceBin'] = np.where(beh['Confidence'] > beh['medianConf'], 'high', 'low')
#     beh.loc[beh['Confidence'].isna(), 'ConfidenceBin'] = np.nan
    beh.loc[~np.isnan(beh['Probability']), 'ProbabilityBin'] = np.where(
          sp.stats.zscore(
                beh.loc[~np.isnan(beh['Probability']), 'ProbabilityTfmd'], 
                    nan_policy='omit') >= 0, 'high', 'low')
#     conditions = [(beh['Choice2'] == 'left') & (beh['StimLeftFreq'] == 60),
#                   (beh['Choice2'] == 'left') & (beh['StimLeftFreq'] == 64),
#                   (beh['Choice2'] == 'right') & (beh['StimRightFreq'] == 60),
#                   (beh['Choice2'] == 'right') & (beh['StimRightFreq'] == 64)]
#     choices = ["Attend_60L", "Attend_64L", "Attend_60R", "Attend_64R"]
#     beh['ChoiceSubtraction'] = np.select(conditions, choices, default="nan")
    conditions = [(beh['HPL'] == 'left') & (beh['StimLeftFreq'] == 60),
                  (beh['HPL'] == 'left') & (beh['StimLeftFreq'] == 64),
                  (beh['HPL'] == 'right') & (beh['StimRightFreq'] == 60),
                  (beh['HPL'] == 'right') & (beh['StimRightFreq'] == 64)]
    choices = ["Attend_60L", "Attend_64L", "Attend_60R", "Attend_64R"]
    beh['HPLSubtraction'] = np.select(conditions, choices, default="nan")

    
    ### Note: Run if analysing post-target time period to exclude trails which we also exclude from the behavioural analysis
    ## extract trial number of trials to exlcude 
    filt_idx = beh[(beh['RT'] == 999) | (beh['RT'] < 0.2)]['RunningTrialNo2'].to_list() 

    ### Find bad epochs
    ## load bad epochs txt file
    bad_epochs_paths = sorted(Path(DERIVATIVES_DATA_PATH).glob(f'**/{subID}_bad*epochs.txt')) 
    ## get bad eeg epochs
    with open(bad_epochs_paths[0], 'r') as file:
        eeg_bads_str = file.readlines()
    eeg_bads = re.findall(r"-?\d*\.?\d+", eeg_bads_str[0])
    eeg_bads = list(map(int, eeg_bads)) # convert to list of ints
    ## get bad eye epochs - eye tracking data was really bad, so didn't bother excluding these
    # with open(bad_epochs_paths[1], 'r') as file:
    #     eye_bads_str = file.readlines()
    # eye_bads = re.findall(r"-?\d*\.?\d+", eye_bads_str[0])
    # eye_bads = list(map(int, eye_bads)) # convert to list of ints

    ### Note: which epochs are excluded will depend on whether you are looking at the pre- or post-target interval (i.e., do you want to also exclude trials based on behaviour)
    ## combine bad epochs with list of trials to exclude, remove duplicates
    exclude = list(set(filt_idx + eeg_bads)) # eye_bads
    #exclude = eeg_bads
    #exclude = eye_bads
    ## drop epochs 
    epochs.drop(exclude)
    ## remove corresponding epochs from the behavioural data file
    beh_cleaned = beh[~beh['RunningTrialNo2'].isin(exclude)]
    ## add new column to behavioural data file with new index
    beh_cleaned['NewTrialIndex'] = np.arange(0, len(beh_cleaned))

    ### Start coherence analysis ----------------------------------------------------------------------
    ## find indexs of trials to group together
    idx_dict = {}

    ## Note: change labels here for different groupings

      # State    
#     for i in ['Attend_60L', 'Attend_60R', 'Attend_64L', 'Attend_64R']:
#           for j in ['low', 'medLow', 'medHigh', 'high']:
#                idx_dict[f'{i}_{j}_State'] = beh_cleaned[(beh_cleaned['HPLSubtraction'] == i) &
#                                                         (beh_cleaned['State'] == j)
#                                                          ]['NewTrialIndex'].values
#     # Confidence 
    for i in ['Attend_60L', 'Attend_60R', 'Attend_64L', 'Attend_64R']:
          for j in ['high', 'low']:
               idx_dict[f'{i}_{j}_ConfidenceBin'] = beh_cleaned[
                    (beh_cleaned['HPLSubtraction'] == i) &
                    (beh_cleaned['ConfidenceBin']== j) &
                    (beh_cleaned['ChoiceHPLCongruent'] == 'Congruent')
                    ]['NewTrialIndex'].values

    ## Crop epochs -----------------------------------------------------------------------------------------

## The diagram below shows how the epochs are cropped for the different analyses ##
#
# Original epochs (3.8 s): 
#
#                    1s      0.67s  0.67s    0.866s
#   |----------|------------|-----|-----|------------|-------------| 
# -0.8         0           1.3   1.367 1.434        2.3            3
#            PLH1          TAR   MSK   PLH2        Max RT
#
#
# Pre-target period analysis (2.8 s):
#
#            1.3s Analysis window
#              ______________  
#   |----------|------------|-----|-----|---------| 
# -0.8         0           1.3   1.367 1.434      2
#
# Post-target period analysis (3.7 s):
#                   
#                         2.3s Analysis Window
#              _______________________________________
#   |----------|------------|-----|-----|------------|------------|
# -0.8         0           1.3   1.367 1.434        2.3          2.9

# ------------------------------------------------------------------------------------------------------------

    ## create a copy of epoched data
    #epochs_coh = epochs.crop(tmax=2.1).pick('eeg') # pre-target period
    epochs_coh = epochs.copy() # post-target

    ## correct for phase offset
    epochs_coh_data = epochs_coh.get_data() # get data as array
    phase_offset_60Hz = 8 - beh_cleaned['60HzPhaseOffset'].to_numpy() # find how many samples of the generating 60 Hz sine wave to adjust by (8 samples is 1 full cycle of a 60 Hz wave sampled at 480 Hz refresh rate)
    samples_per_frame = (1000/480)/(1000/2048) # EEG samples per frame: 1000 ms, 480 Hz PROPixx refresh rate, 2048 Hz EEG sample rate
    phase_offset_samples = np.round(phase_offset_60Hz*samples_per_frame).astype(int) # convert samples of sine wave into EEG samples
    epochs_coh_po = [epochs_coh_data[i, :, j:] 
                         for i, j in enumerate(phase_offset_samples)] # trim data from start by appropriate number of EEG samples
    
    # make all epochs_coh_po the same length after account for phase offset
    # pre-target period
    #epochs_coh_po = [epoch[:, 0:5735] ## make epochs 2.8 seconds in length: -0.8 - 2
    #                       for epoch in epochs_coh_po] # need to trim this per epoch as it's stored as a list
    # post-target period
    end = epochs_coh_data.shape[-1] - int(np.round(0.1*2048)) # trim all epochs by 0.1 s
    epochs_coh_po = [epoch[:, 0:end] 
                             for epoch in epochs_coh_po] # need to trim this per epoch as it's stored as a list
    
    ## Turn trimmed data into epochs class
    epochs_coh_po_data = np.array(epochs_coh_po)
    info = mne.create_info(epochs_coh.ch_names, 
                         epochs_coh.info['sfreq'], 
                         ch_types='eeg')
     # Pre-target period
#     epochs_coh_po = mne.EpochsArray(epochs_coh_po_data, 
#                                 info, 
#                                 tmin=-0.8, 
#                                 event_id=None)
#     epochs_coh = epochs_coh.crop(tmax=2).pick('eeg')
    # Post-target period
    epochs_coh_po = mne.EpochsArray(epochs_coh_po_data, 
                                info, 
                                tmin=-0.8,
                                event_id=None).pick('eeg')
    epochs_coh = epochs_coh.crop(tmax=2.9).pick('eeg') 
    
    ## Compute and save coherence output
    file = f'{DERIVATIVES_DATA_PATH}processed/{subID}_coherence_allTrials_tar-long.hdf5'
    f = h5.File(file, 'a')

    # condition-specific analyses at 60 and 64 Hz
#     for condition in idx_dict.keys():
#         trl_idxs = idx_dict[condition]
#         if len(trl_idxs) == 0:
#              f[f'{condition}_coh'] = np.empty(0)
#              f[f'{condition}_cohpo'] = np.empty(0)
#         else:
#              f[f'{condition}_coh'] = compute_coherence(epochs_coh[trl_idxs], [64]) 
#              f[f'{condition}_cohpo'] = compute_coherence(epochs_coh_po[trl_idxs], [60]) 
#     f.close()

    #  spectrogram plot
#     f = h5.File(file, 'a')
#     f['64_coh'] = compute_coherence(epochs_coh, np.arange(56, 69)) # [64]
#     f['60_cohpo'] = compute_coherence(epochs_coh_po, np.arange(56, 69)) #[60]
#     f.close()



### Average over all subjects to find electrodes to look at ### 
cohDataEEG = sorted(Path(DERIVATIVES_DATA_PATH).glob('**/*coherence_allTrials_tar-long.hdf5'))
exclude = ["sub-05", "sub-10", "sub-21", "sub-24", "sub-37", "sub-40"] # from behavioural accuracy data

with h5.File(cohDataEEG[0], 'r') as f:
     shape = f[list(f.keys())[1]].shape # get shape of data file, should be same across all pts

n_subs = 40 - len(exclude)
n_tagfeqs = 2 # 60Hz and 64Hz
n_specfreqs = shape[0] # plot coherence at these frequences
n_elec = shape[1] # how many electrodes we have data for
n_tpoints = shape[2] # how man timepoints we have data for

# create empty array to store data
sub_count = 0
coh_data_all_subs = np.empty([n_subs, n_tagfeqs, n_specfreqs, n_elec, n_tpoints])
for path in cohDataEEG:

    subID, _, _, _= path.stem.split("_") 

    if subID in exclude:
         continue

    f = h5.File(path, 'r')
    d_60 = f['60_cohpo'][:, 0:64, :]
    d_64 = f['64_coh'][:]
    coh_data_all_subs[sub_count, 0] = d_60 #np.squeeze(d_60, 0)
    coh_data_all_subs[sub_count, 1] = d_64 #np.squeeze(d_64, 0)

    f.close()
    sub_count += 1

freq_idxs = np.isin(np.arange(56, 69), [60, 64])
#coh_av = np.mean(coh_data_all_subs[:, :, freq_idxs, :, :], (0, 1, 2))
coh_av_60 = np.mean(coh_data_all_subs[:, 0, 4, :, :], (0)) # extract 60 Hz data and average over subjects: index all subs, the 60 Hz data, the 60 Hz frequency, all electrodes and all time points 
coh_av_64 = np.mean(coh_data_all_subs[:, 1, 8, :, :], (0)) # extract 64 Hz data and average over subjects: index all subs, the 64 Hz data, the 64 Hz frequency, all electrodes and all time points
coh_av = (coh_av_60 + coh_av_64) / 2

# identify time window of interest
start = int(0.8*2048) # 0.7
end = int(2.1*2048) # 2.9
# find channels with highest averaged coherence over the placeholder period
coh_av_chan = coh_av[:, start:end].mean(1)
chan_idxs = np.flip(np.argsort(coh_av_chan))[0:6] # take top 6
montage = mne.channels.make_standard_montage('biosemi64')
chan_names = np.array(montage.ch_names)[chan_idxs] 
chan_names = ['POz', 'P2', 'Pz', 'PO4', 'P1', 'P4']
chan_bool = np.isin(np.array(montage.ch_names), chan_names)

temp_info = mne.create_info(ch_names=montage.ch_names, sfreq=2048, ch_types='eeg')
evoked_60 = mne.EvokedArray(coh_av_60, temp_info, tmin=-0.4).set_montage(montage)
evoked_60.plot(scalings=dict(eeg=1), units='au')
evoked_60.plot_topomap(scalings=dict(eeg=1), units='au', times=[-0.1, 0.2, 1])
evoked_60.plot_joint(times=[-0.1, 0.3, 1])

evoked_64 = mne.EvokedArray(coh_av_64, temp_info, tmin=-0.4).set_montage(montage)
evoked_64.plot(scalings=dict(eeg=1), units='au')
evoked_64.plot_topomap(scalings=dict(eeg=1), units='au', times=[-0.1, 0.2, 1])
evoked_64.plot_joint(times=[-0.1, 0.3, 1])

evoked_all = mne.EvokedArray(coh_av, temp_info, tmin=-0.8).set_montage(montage)
evoked_all.plot(scalings=dict(eeg=1), units='au')
fig = evoked_all.plot_topomap(scalings=dict(eeg=1), units='au', times=[-0.1, 0.0, 1, 1.3], vlim=[0, None], contours=False, cmap='plasma')
evoked_all.plot_joint(times=[-0.1, 0.0, 1, 1.3])

fig.savefig('C:\\Users\\cstone\\OneDrive - UNSW\\Documents\\Presentations\\NeuroAlliance\\topo.svg', dpi=400)


datasets = [np.flipud(coh_data_all_subs[:, 0, :, chan_idxs, 1434:5940].mean((0, 1))), 
            np.flipud(coh_data_all_subs[:, 1, :, chan_idxs, 1434:5940].mean((0, 1)))]

fig, axs = plt.subplots(1, 2)
fig.suptitle('Coherence')
norm = matplotlib.colors.Normalize(vmin=0, vmax=0.1) #vmin=np.min(datasets), vmax=np.max(datasets
images = []
for ax, data in zip(axs.flat, datasets):
    images.append(ax.imshow(data, aspect='auto', norm=norm, cmap='plasma'))
    ax.set_xticks([205, 1229, 2253, 3277, 4301])
    ax.set_xticklabels(['0.0', '0.5', '1.0',  '1.5', '2.0'])
    ax.set_xlabel('Time since placeholder (s)')
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_yticklabels([68, 66, 64, 62, 60, 58, 56])
    ax.set_ylabel('Frequency (Hz)')
fig.colorbar(images[1], ax=axs, orientation='horizontal', fraction=.1)
fig.show()

fig.savefig('C:\\Users\\cstone\\OneDrive - UNSW\\Documents\\Presentations\\NeuroAlliance\\spect2.svg', dpi=400)


### Compute coherence differences with identified channels ###
## State 
cohDataEEG = sorted(Path(DERIVATIVES_DATA_PATH).glob('**/*State_HPLcongruent_tar-long.hdf5'))
HIGH = []
MEDHIGH = []
MEDLOW = []
LOW = []

for path in cohDataEEG:

    f = h5.File(path, 'r')

    ## High 
    # power difference at 60 Hz when attending/not attending 60 Hz on left
    high_A60L = f['Attend_60L_high_State_cohpo'][:, 0:64, :]
    high_A64R = f['Attend_64R_high_State_cohpo'][:, 0:64, :]
    high_60L = high_A60L - high_A64R
    # power difference at 60 Hz when attending/not attending 60 Hz on right
    high_A60R = f['Attend_60R_high_State_cohpo'][:, 0:64, :]
    high_A64L = f['Attend_64L_high_State_cohpo'][:, 0:64, :]
    high_60R = high_A60R - high_A64L
    # average over left and right hemifields
    high_60 = (high_60L + high_60R)/2

    # power difference at 64 Hz when attending/not attending 64 Hz on left
    high_A64L = f['Attend_64L_high_State_coh'][:] 
    high_A60R = f['Attend_60R_high_State_coh'][:]
    high_64L = high_A64L - high_A60R
    # power difference at 64 Hz when attending/not attending 64 Hz on right
    high_A64R = f['Attend_64R_high_State_coh'][:]
    high_A60L = f['Attend_60L_high_State_coh'][:]
    high_64R = high_A64R - high_A60L
    # average over left and right hemifields
    high_64 = (high_64L + high_64R)/2
    # average over frequencies
    high = (high_60 + high_64)/2

    ## medHigh
    # power difference at 60 Hz when attending/not attending 60 Hz on left
    medHigh_A60L = f['Attend_60L_medHigh_State_cohpo'][:, 0:64, :]
    medHigh_A64R = f['Attend_64R_medHigh_State_cohpo'][:, 0:64, :]
    medHigh_60L = medHigh_A60L - medHigh_A64R
    # power difference at 60 Hz when attending/not attending 60 Hz on right
    medHigh_A60R = f['Attend_60R_medHigh_State_cohpo'][:, 0:64, :]
    medHigh_A64L = f['Attend_64L_medHigh_State_cohpo'][:, 0:64, :]
    medHigh_60R = medHigh_A60R - medHigh_A64L
    # average over left and right hemifields
    medHigh_60 = (medHigh_60L + medHigh_60R)/2

    # power difference at 64 Hz when attending/not attending 64 Hz on left
    medHigh_A64L = f['Attend_64L_medHigh_State_coh'][:]
    medHigh_A60R = f['Attend_60R_medHigh_State_coh'][:]
    medHigh_64L = medHigh_A64L - medHigh_A60R
    # power difference at 64 Hz when attending/not attending 64 Hz on right
    medHigh_A64R = f['Attend_64R_medHigh_State_coh'][:]
    medHigh_A60L = f['Attend_60L_medHigh_State_coh'][:]
    medHigh_64R = medHigh_A64R - medHigh_A60L
    # average over left and right hemifields
    medHigh_64 = (medHigh_64L + medHigh_64R)/2
    # average over frequencies
    medHigh = (medHigh_60 + medHigh_64)/2

    ## medLow
    # power difference at 60 Hz when attending/not attending 60 Hz on left
    medLow_A60L = f['Attend_60L_medLow_State_cohpo'][:, 0:64, :]
    medLow_A64R = f['Attend_64R_medLow_State_cohpo'][:, 0:64, :]
    medLow_60L = medLow_A60L - medLow_A64R
    # power difference at 60 Hz when attending/not attending 60 Hz on right
    medLow_A60R = f['Attend_60R_medLow_State_cohpo'][:, 0:64, :]
    medLow_A64L = f['Attend_64L_medLow_State_cohpo'][:, 0:64, :]
    medLow_60R = medLow_A60R - medLow_A64L
    # average over left and right hemifields
    medLow_60 = (medLow_60L + medLow_60R)/2

    # power difference at 64 Hz when attending/not attending 64 Hz on left
    medLow_A64L = f['Attend_64L_medLow_State_coh'][:] 
    medLow_A60R = f['Attend_60R_medLow_State_coh'][:]
    medLow_64L = medLow_A64L - medLow_A60R
    # power difference at 64 Hz when attending/not attending 64 Hz on right
    medLow_A64R = f['Attend_64R_medLow_State_coh'][:]
    medLow_A60L = f['Attend_60L_medLow_State_coh'][:]
    medLow_64R = medLow_A64R - medLow_A60L
    # average over left and right hemifields
    medLow_64 = (medLow_64L + medLow_64R)/2
    # average over frequencies
    medLow = (medLow_60 + medLow_64)/2

    ## Low State, Low ConfidenceBin
    # power difference at 60 Hz when attending/not attending 60 Hz on left
    low_A60L = f['Attend_60L_low_State_cohpo'][:, 0:64, :]
    low_A64R = f['Attend_64R_low_State_cohpo'][:, 0:64, :]
    low_60L = low_A60L - low_A64R
    # power difference at 60 Hz when attending/not attending 60 Hz on right
    low_A60R = f['Attend_60R_low_State_cohpo'][:, 0:64, :]
    low_A64L = f['Attend_64L_low_State_cohpo'][:, 0:64, :]
    low_60R = low_A60R - low_A64L
    # average over left and right hemifields
    low_60 = (low_60L + low_60R)/2

    # power difference at 64 Hz when attending/not attending 64 Hz on left
    low_A64L = f['Attend_64L_low_State_coh'][:] 
    low_A60R = f['Attend_60R_low_State_coh'][:]
    low_64L = low_A64L - low_A60R
    # power difference at 64 Hz when attending/not attending 64 Hz on right
    low_A64R = f['Attend_64R_low_State_coh'][:]
    low_A60L = f['Attend_60L_low_State_coh'][:]
    low_64R = low_A64R - low_A60L
    # average over left and right hemifields
    low_64 = (low_64L + low_64R)/2
    # average over frequencies
    low = (low_60 + low_64)/2

    # save output
    HIGH.append(high)
    MEDHIGH.append(medHigh)
    MEDLOW.append(medLow)
    LOW.append(low)
    f.close()

# export data for analysis
columns = [f'T{t}' for t in range(HIGH[0].shape[-1])]
dfs = [pd.DataFrame(data=np.array(df)[:, :, chan_bool, :].mean((1, 2)),
                    columns=columns) 
       for df in [HIGH, MEDHIGH, MEDLOW, LOW]]
df = pd.concat(dfs, axis=0)
df['Condition'] = np.repeat(['HIGH', 'MEDHIGH', 'MEDLOW', 'LOW'], repeats=40)
df['Subject'] = [f'sub-{sub}' for sub in range(1, 41)]*4
df.to_csv(DERIVATIVES_DATA_PATH + 'processed/Honours2026_eeg_coherence_State_tar-long.csv')


## Confidence 
cohDataEEG = sorted(Path(DERIVATIVES_DATA_PATH).glob('**/*Confidence_HPLcongruent_cong-tar-long.hdf5'))
HIGH = []
LOW = []

for path in cohDataEEG:


    f = h5.File(path, 'r')

    ## High 
    # power difference at 60 Hz when attending/not attending 60 Hz on left
    high_A60L = f['Attend_60L_high_ConfidenceBin_cohpo'][:][:, 0:64, :]
    high_A64R = f['Attend_64R_high_ConfidenceBin_cohpo'][:][:, 0:64, :]
    high_60L = high_A60L - high_A64R
    # power difference at 60 Hz when attending/not attending 60 Hz on right
    high_A60R = f['Attend_60R_high_ConfidenceBin_cohpo'][:][:, 0:64, :]
    high_A64L = f['Attend_64L_high_ConfidenceBin_cohpo'][:][:, 0:64, :]
    high_60R = high_A60R - high_A64L
    # average over left and right hemifields
    high_60 = (high_60L + high_60R)/2

    # power difference at 64 Hz when attending/not attending 64 Hz on left
    high_A64L = f['Attend_64L_high_ConfidenceBin_coh'][:] 
    high_A60R = f['Attend_60R_high_ConfidenceBin_coh'][:]
    high_64L = high_A64L - high_A60R
    # power difference at 64 Hz when attending/not attending 64 Hz on right
    high_A64R = f['Attend_64R_high_ConfidenceBin_coh'][:]
    high_A60L = f['Attend_60L_high_ConfidenceBin_coh'][:]
    high_64R = high_A64R - high_A60L
    # average over left and right hemifields
    high_64 = (high_64L + high_64R)/2
    # average over frequencies
    high = (high_60 + high_64)/2

    ## Low
    # power difference at 60 Hz when attending/not attending 60 Hz on left
    low_A60L = f['Attend_60L_low_ConfidenceBin_cohpo'][:][:, 0:64, :]
    low_A64R = f['Attend_64R_low_ConfidenceBin_cohpo'][:][:, 0:64, :]
    low_60L = low_A60L - low_A64R
    # power difference at 60 Hz when attending/not attending 60 Hz on right
    low_A60R = f['Attend_60R_low_ConfidenceBin_cohpo'][:][:, 0:64, :]
    low_A64L = f['Attend_64L_low_ConfidenceBin_cohpo'][:][:, 0:64, :]
    low_60R = low_A60R - low_A64L
    # average over left and right hemifields
    low_60 = (low_60L + low_60R)/2

    # power difference at 64 Hz when attending/not attending 64 Hz on left
    low_A64L = f['Attend_64L_low_ConfidenceBin_coh'][:] 
    low_A60R = f['Attend_60R_low_ConfidenceBin_coh'][:]
    low_64L = low_A64L - low_A60R
    # power difference at 64 Hz when attending/not attending 64 Hz on right
    low_A64R = f['Attend_64R_low_ConfidenceBin_coh'][:]
    low_A60L = f['Attend_60L_low_ConfidenceBin_coh'][:]
    low_64R = low_A64R - low_A60L
    # average over left and right hemifields
    low_64 = (low_64L + low_64R)/2
    # average over frequencies
    low = (low_60 + low_64)/2

    # save output
    HIGH.append(high)
    LOW.append(low)
    f.close()

# export data for analysis
columns = [f'T{t}' for t in range(HIGH[0].shape[-1])]
dfs = [pd.DataFrame(data=np.array(df)[:, :, chan_bool, :].mean((1, 2)),
                    columns=columns) 
       for df in [HIGH, LOW]]
df = pd.concat(dfs, axis=0)
df.reset_index(inplace=True)

df_ids = pd.DataFrame()
df_ids['Condition'] = np.repeat(['HIGH', 'LOW'], repeats=40)
df_ids['Subject'] = [f'sub-{sub:02d}' for sub in range(1, 41)]*2
#df_ids = df_ids[~df_ids["Subject"].isin(ignore)]
df_ids.reset_index(inplace=True)

df = pd.concat([df_ids, df], axis=1)

df.to_csv(DERIVATIVES_DATA_PATH + 'processed/Honours2026_eeg_coherence_Confidence_cong-tar-long.csv')
