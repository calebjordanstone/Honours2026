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

# epochs = mne.read_epochs(DERIVATIVES_DATA_PATH + "preprocessed/sub-04/sub-04_task-Honours2026_eeg_epochs-plh.fif")
# beh = pl.read_csv(SOURCEDATA_DATA_PATH + "sub-04/beh/sub-04_task-Honours2026_beh.txt", separator='\t')

# run analysis
''' Run coherence using function defined here: https://osf.io/yrh6v/files/gz65u'''

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


for path in srcDataEEG[1:2]:

    # get subID   
    subID = path.stem.split('_')[0]
    # load data 
    epochs = mne.read_epochs(path) 
    beh_path = sorted(Path(SOURCEDATA_DATA_PATH).glob(f'**/{subID}_task-Honours2026_beh.txt'))
    beh = pd.read_csv(beh_path[0], sep='\t')
    # create new varaibles
    beh['Ratio'] = np.where(beh['TargetRatio'].isin(['[0.9, 0.1]', '[0.8, 0.2]']), 'high', 'low')
    beh['Experience'] = np.where(beh['StateTrialNo'] <= 60, 'low', 'high')
    #beh['ConfidenceTfmd'] = beh['Confidence']*0.5 + 0.5
    #beh['ProbabilityTfmd'] = beh['Probability']*0.5 + 0.5
    beh['TargetHPLCongruency'] = np.where(beh['TargetSide'] == beh['HPL'], 'Congruent', 'Incongruent')
    conditions = [(beh['TargetSide'] == 'right') & (beh['Choice'] == 'right'),
                  (beh['TargetSide'] == 'right') & (beh['Choice'] == 'left'),
                  (beh['TargetSide'] == 'left') & (beh['Choice'] == 'right'),
                  (beh['TargetSide'] == 'left') & (beh['Choice'] == 'left')]
    choices = ["Congruent", "Incongruent", "Incongruent", "Congruent"]
    beh["TargetChoiceCongruency"] = np.select(conditions, choices, default="nan")
    beh.loc[~np.isnan(beh['Confidence']), 'ConfidenceBin'] = np.where(
          sp.stats.zscore(
                beh.loc[~np.isnan(beh['Confidence']), 'Confidence'], 
                nan_policy='omit') >= 0, 'high', 'low')
    beh.loc[~np.isnan(beh['Probability']), 'ProbabilityBin'] = np.where(
          sp.stats.zscore(
                beh.loc[~np.isnan(beh['Probability']), 'Probability'], 
                nan_policy='omit') >= 0, 'high', 'low')
    # conditions = [(beh['HPL'] == 'left') & (beh['StimLeftFreq'] == 60),
    #               (beh['HPL'] == 'left') & (beh['StimLeftFreq'] == 64),
    #               (beh['HPL'] == 'right') & (beh['StimRightFreq'] == 60),
    #               (beh['HPL'] == 'right') & (beh['StimRightFreq'] == 64)]
    # choices = ["60-64", "64-60", "60-64", "64-60"]
    # beh['HPLSubtraction'] = np.select(conditions, choices, default="nan")
    # conditions = [(beh['Choice'] == 'left') & (beh['StimLeftFreq'] == 60),
    #               (beh['Choice'] == 'left') & (beh['StimLeftFreq'] == 64),
    #               (beh['Choice'] == 'right') & (beh['StimRightFreq'] == 60),
    #               (beh['Choice'] == 'right') & (beh['StimRightFreq'] == 64)]
    # choices = ["60-64", "64-60", "60-64", "64-60"]
    # beh['ChoiceSubtraction'] = np.select(conditions, choices, default="nan")
    conditions = [(beh['HPL'] == 'left') & (beh['StimLeftFreq'] == 60),
                  (beh['HPL'] == 'left') & (beh['StimLeftFreq'] == 64),
                  (beh['HPL'] == 'right') & (beh['StimRightFreq'] == 60),
                  (beh['HPL'] == 'right') & (beh['StimRightFreq'] == 64)]
    choices = ["Attend_60L", "Attend_64L", "Attend_60R", "Attend_64R"]
    beh['HPLSubtraction'] = np.select(conditions, choices, default="nan")

    # extract trial number of trials to exlcude
    #     filt_idx = beh.filter((pl.col('Accuracy') != 1) | 
    #                           (pl.col('RT') < 0.15)
    #                           ).select('RunningTrialNo2').to_series().to_list() ## think about these
    filt_idx = []
    # load bad epochs txt file
    bad_epochs_path = sorted(Path(DERIVATIVES_DATA_PATH).glob(f'**/{subID}_bad_epochs.txt')) 
    with open(bad_epochs_path[0], 'r') as file:
        bads_str = file.readlines()
    bads = re.findall(r"-?\d*\.?\d+", bads_str[0])
    bads = list(map(int, bads)) # convert to list of ints

    # combine bad epochs with list of trials to exclude, remove duplicates
    exclude = list(set(filt_idx + bads)) 
    # drop epochs 
    epochs.drop(exclude)
    # remove corresponding epochs from the behavioural data file
    beh['RunningTrialNo2'] = beh['RunningTrialNo'] - 1 # re-set running trial number to account for python indexing starting at 0
    beh_cleaned = beh[~beh['RunningTrialNo2'].isin(exclude)]
    # add new column to behavioural data file with new index
    beh_cleaned['NewTrialIndex'] = np.arange(0, len(beh_cleaned))

    # start coherence analysis ----------------------------------------------------------------------
    # find indexs of trials to group together
    idx_dict = {}
    for i in ['Attend_60L', 'Attend_60R', 'Attend_64L', 'Attend_64R']:
          for j in ['high', 'low']:
                for k in ['high', 'low']:
                    idx_dict[f'{i}_{j}_ratio_{k}_experience'] = beh[(beh['HPLSubtraction'] == i) &
                                                                    (beh['Ratio'] == j) &
                                                                    (beh['Experience'] == k)
                                                                    ]['NewTrialIndex'].values
    # crop epochs
    epochs_coh = epochs.copy().crop(tmin=-0.4, tmax=1.7).pick('eeg')
    # correct for phase offset
    epochs_coh_data = epochs_coh.get_data()
    phase_offset_60Hz = 8 - beh_cleaned['60HzPhaseOffset'].to_numpy()
    samples_per_frame = (1000/480)/(1000/2048) # 1000 ms, 480 Hz PROPixx refresh rate, 2048 Hz EEG sample rate
    phase_offset_samples = np.round(phase_offset_60Hz*samples_per_frame).astype(int)
    epochs_coh_po = [epochs_coh_data[i, :, j:-1] 
                         for i, j in enumerate(phase_offset_samples)]
    #end = epochs_coh_data.shape[-1] - np.max(phase_offset_samples)
    epochs_coh_po = [epoch[:, 0:4096] ## make epochs 2 seconds in length: -0.4 - 1.6
                         for epoch in epochs_coh_po] # need to trim this per epoch as it's stored as a list
    epochs_coh_po_data = np.array(epochs_coh_po)
    info = mne.create_info(epochs_coh.ch_names, 
                         epochs_coh.info['sfreq'], 
                         ch_types='eeg')
    epochs_coh_po = mne.EpochsArray(epochs_coh_po_data, 
                                info, 
                                tmin=0.0, 
                                event_id=None)
    epochs_coh = epochs_coh.crop(tmin=-0.4, tmax=1.6).pick('eeg') # make same length as epochs_coh_po
    # compute and save coherence output
    file = f'{DERIVATIVES_DATA_PATH}processed/{subID}_coherence.hdf5'
    f = h5.File(file, 'a')
    for condition in idx_dict.keys():
        trl_idxs = idx_dict[condition]
        f[f'{condition}_coh'] = compute_coherence(epochs_coh[trl_idxs], [64]) 
        f[f'{condition}_cohpo'] = compute_coherence(epochs_coh_po[trl_idxs], [60]) 
    f.close()

### Figure out how to average over all subjects to find electrodes to look at ### Test the below?
cohDataEEG = sorted(Path(DERIVATIVES_DATA_PATH).glob('**/*.hdf5'))
nsubs = 1
coh_data_all_subs = np.empty([nsubs, 32, 64, 4096]) # nsubs, 32 conditions, 64 electrodes, 4096 timepoints
for i, path in enumerate(cohDataEEG):
    f = h5.File(path, 'r')
    for j, key in enumerate(f.keys()):
        d = f[key][:]
        coh_data_all_subs[i, j] = np.squeeze(d, 0)
    f.close()
coh_av = coh_data_all_subs.mean([0, 1])
# identify start and end of placeholder period
start = int(0.4*2048)
end = int(1.7*2048)
# find channels with highest averaged coherence over the placeholder period
coh_av_chan = coh_av[:, :, start:end].mean(2)
chan_idxs = np.flip(np.argsort(coh_av_chan))[0, 0:6] # take top 6
chan_names = np.array(epochs_coh.ch_names)[chan_idxs] 

## Compute coherence differences with identified channels
cohDataEEG = sorted(Path(DERIVATIVES_DATA_PATH).glob('**/*.hdf5'))
HR_HE_all = []
HR_LE_all = []
LR_HE_all = []
LR_LE_all = []
for path in cohDataEEG:
    f = h5.File(path, 'r')

    ## High Ratio, High Experience
    # power difference at 60 Hz when attending/not attending 60 Hz on left
    HR_HE_A60L = f['Attend_60L_high_ratio_high_experience_cohpo'][:]
    HR_HE_A64R = f['Attend_64R_high_ratio_high_experience_cohpo'][:]
    HR_HE_60L = HR_HE_A60L - HR_HE_A64R
    # power difference at 60 Hz when attending/not attending 60 Hz on right
    HR_HE_A60R = f['Attend_60R_high_ratio_high_experience_cohpo'][:]
    HR_HE_A64L = f['Attend_64L_high_ratio_high_experience_cohpo'][:]
    HR_HE_60R = HR_HE_A60R - HR_HE_A64L
    # average over left and right hemifields
    HR_HE_60 = (HR_HE_60L + HR_HE_60R)/2

    # power difference at 64 Hz when attending/not attending 64 Hz on left
    HR_HE_A64L = f['Attend_64L_high_ratio_high_experience_coh'][:, :, 0:-1] # these trials have 1 extra time point 
    HR_HE_A60R = f['Attend_60R_high_ratio_high_experience_coh'][:, :, 0:-1]
    HR_HE_64L = HR_HE_A64L - HR_HE_A60R
    # power difference at 64 Hz when attending/not attending 64 Hz on right
    HR_HE_A64R = f['Attend_64R_high_ratio_high_experience_coh'][:, :, 0:-1]
    HR_HE_A60L = f['Attend_60L_high_ratio_high_experience_coh'][:, :, 0:-1]
    HR_HE_64R = HR_HE_A64R - HR_HE_A60L
    # average over left and right hemifields
    HR_HE_64 = (HR_HE_64L + HR_HE_64R)/2
    # average over frequencies
    HR_HE = (HR_HE_60 + HR_HE_64)/2

    ## High Ratio, Low Experience
    # power difference at 60 Hz when attending/not attending 60 Hz on left
    HR_LE_A60L = f['Attend_60L_high_ratio_low_experience_cohpo'][:]
    HR_LE_A64R = f['Attend_64R_high_ratio_low_experience_cohpo'][:]
    HR_LE_60L = HR_LE_A60L - HR_LE_A64R
    # power difference at 60 Hz when attending/not attending 60 Hz on right
    HR_LE_A60R = f['Attend_60R_high_ratio_low_experience_cohpo'][:]
    HR_LE_A64L = f['Attend_64L_high_ratio_low_experience_cohpo'][:]
    HR_LE_60R = HR_LE_A60R - HR_LE_A64L
    # average over left and right hemifields
    HR_LE_60 = (HR_LE_60L + HR_LE_60R)/2

    # power difference at 64 Hz when attending/not attending 64 Hz on left
    HR_LE_A64L = f['Attend_64L_high_ratio_low_experience_coh'][:, :, 0:-1] # these trials have 1 extra time point 
    HR_LE_A60R = f['Attend_60R_high_ratio_low_experience_coh'][:, :, 0:-1]
    HR_LE_64L = HR_LE_A64L - HR_LE_A60R
    # power difference at 64 Hz when attending/not attending 64 Hz on right
    HR_LE_A64R = f['Attend_64R_high_ratio_low_experience_coh'][:, :, 0:-1]
    HR_LE_A60L = f['Attend_60L_high_ratio_low_experience_coh'][:, :, 0:-1]
    HR_LE_64R = HR_LE_A64R - HR_LE_A60L
    # average over left and right hemifields
    HR_LE_64 = (HR_LE_64L + HR_LE_64R)/2
    # average over frequencies
    HR_LE = (HR_LE_60 + HR_LE_64)/2

    ## Low Ratio, High Experience
    # power difference at 60 Hz when attending/not attending 60 Hz on left
    LR_HE_A60L = f['Attend_60L_low_ratio_high_experience_cohpo'][:]
    LR_HE_A64R = f['Attend_64R_low_ratio_high_experience_cohpo'][:]
    LR_HE_60L = LR_HE_A60L - LR_HE_A64R
    # power difference at 60 Hz when attending/not attending 60 Hz on right
    LR_HE_A60R = f['Attend_60R_low_ratio_high_experience_cohpo'][:]
    LR_HE_A64L = f['Attend_64L_low_ratio_high_experience_cohpo'][:]
    LR_HE_60R = LR_HE_A60R - LR_HE_A64L
    # average over left and right hemifields
    LR_HE_60 = (LR_HE_60L + LR_HE_60R)/2

    # power difference at 64 Hz when attending/not attending 64 Hz on left
    LR_HE_A64L = f['Attend_64L_low_ratio_high_experience_coh'][:, :, 0:-1] # these trials have 1 extra time point 
    LR_HE_A60R = f['Attend_60R_low_ratio_high_experience_coh'][:, :, 0:-1]
    LR_HE_64L = LR_HE_A64L - LR_HE_A60R
    # power difference at 64 Hz when attending/not attending 64 Hz on right
    LR_HE_A64R = f['Attend_64R_low_ratio_high_experience_coh'][:, :, 0:-1]
    LR_HE_A60L = f['Attend_60L_low_ratio_high_experience_coh'][:, :, 0:-1]
    LR_HE_64R = LR_HE_A64R - LR_HE_A60L
    # average over left and right hemifields
    LR_HE_64 = (LR_HE_64L + LR_HE_64R)/2
    # average over frequencies
    LR_HE = (LR_HE_60 + LR_HE_64)/2

    ## Low Ratio, Low Experience
    # power difference at 60 Hz when attending/not attending 60 Hz on left
    LR_LE_A60L = f['Attend_60L_low_ratio_low_experience_cohpo'][:]
    LR_LE_A64R = f['Attend_64R_low_ratio_low_experience_cohpo'][:]
    LR_LE_60L = LR_LE_A60L - LR_LE_A64R
    # power difference at 60 Hz when attending/not attending 60 Hz on right
    LR_LE_A60R = f['Attend_60R_low_ratio_low_experience_cohpo'][:]
    LR_LE_A64L = f['Attend_64L_low_ratio_low_experience_cohpo'][:]
    LR_LE_60R = LR_LE_A60R - LR_LE_A64L
    # average over left and right hemifields
    LR_LE_60 = (LR_LE_60L + LR_LE_60R)/2

    # power difference at 64 Hz when attending/not attending 64 Hz on left
    LR_LE_A64L = f['Attend_64L_low_ratio_low_experience_coh'][:, :, 0:-1] # these trials have 1 extra time point 
    LR_LE_A60R = f['Attend_60R_low_ratio_low_experience_coh'][:, :, 0:-1]
    LR_LE_64L = LR_LE_A64L - LR_LE_A60R
    # power difference at 64 Hz when attending/not attending 64 Hz on right
    LR_LE_A64R = f['Attend_64R_low_ratio_low_experience_coh'][:, :, 0:-1]
    LR_LE_A60L = f['Attend_60L_low_ratio_low_experience_coh'][:, :, 0:-1]
    LR_LE_64R = LR_LE_A64R - LR_LE_A60L
    # average over left and right hemifields
    LR_LE_64 = (LR_LE_64L + LR_LE_64R)/2
    # average over frequencies
    LR_LE = (LR_LE_60 + LR_LE_64)/2

    # save output
    HR_HE_all.append(HR_HE)
    HR_LE_all.append(HR_LE)
    LR_HE_all.append(LR_HE)
    LR_LE_all.append(LR_LE)
    f.close()


## Average and plot
start = int(0.2*2048)
end = int(1.7*2048)
plt.plot(HR_HE[0, 29, start:end], label='HR_HE')
plt.plot(HR_LE[0, 29, start:end], label='HR_LE')
plt.plot(LR_HE[0, 29, start:end], label='LR_HE')
plt.plot(LR_LE[0, 29, start:end], label='LR_LE')
plt.legend()

# epochs_coh_av = mne.EvokedArray(np.squeeze(coh_av, 0), 
#                                 info, 
#                                 tmin=-0.4).set_montage('biosemi64')

# # plot coherence
# fig, ax = plt.subplots()
# im = ax.imshow(coh_po[:, 63, :], aspect='auto') # electrode #29 is POz
# ax.set_xticks([410, 820, 1229, 1639, 2048])
# ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8'])
# ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
# ax.set_yticklabels([56, 58, 60, 62, 64, 66, 68])
# fig.show()
# fig.colorbar(im)

# start = int(0.2*2048)
# end = int(1.7*2048)
# datasets = [coh_po[:, 29, start:end], coh[:, 29, start:end]]
# fig, axs = plt.subplots(1, 2)
# fig.suptitle('Coherence')
# norm = matplotlib.colors.Normalize(vmin=np.min(datasets), vmax=np.max(datasets))
# images = []
# for ax, data in zip(axs.flat, datasets):
#     images.append(ax.imshow(data, norm=norm, aspect='auto'))
#     ax.set_xticks([410, 820, 1229, 1639, 2048])
#     ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8'])
#     ax.set_xlabel('Time since placeholder (s)')
#     ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
#     ax.set_yticklabels([56, 58, 60, 62, 64, 66, 68])
#     ax.set_ylabel('Frequency (Hz)')
# fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)
# fig.show()
