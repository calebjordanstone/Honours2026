import polars as pl
import mne
import numpy as np 
from pathlib import Path
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
%matplotlib qt

## set data paths
DATAPATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/honours_projects/Honours2026/data/sourcedata" 
SAVEPATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/honours_projects/Honours2026/data/derivatives/preprocessed/"
srcDataEEG = sorted(Path(DATAPATH).glob('**/*.bdf'))

for path in srcDataEEG[40:41]:


    # extract some BIDS info
    subID, task, modality = path.stem.split('_')

    # load eye tracking data 
    eye = pl.concat([pl.read_csv(i)  
            for i in sorted(Path(DATAPATH).glob(f'**/{subID}*.csv'))],
            how='vertical_relaxed').sort(by='block', descending=False)
    # define events
    event_dict = {
            # Ratio = 90/10 ('High), HPL = left, Target location = left
            "hi/Le/le/iti": 70,
            "hi/Le/le/plh": 72,
            "hi/Le/le/tar": 74,
            "hi/Le/le/msk": 76,
            "hi/Le/le/fdbc": 78,
            "hi/Le/le/fdbi": 79,
            "hi/Le/le/fdbm": 77,
            # Ratio = 90/10 ('High), HPL = left, Target location = right
            "hi/Le/ri/iti": 80,
            "hi/Le/ri/plh": 82,
            "hi/Le/ri/tar": 84,
            "hi/Le/ri/msk": 86,
            "hi/Le/ri/fdbc": 88,
            "hi/Le/ri/fdbi": 89,
            "hi/Le/ri/fdbm": 87,
            # Ratio = 90/10 ('High), HPL = right, Target location = left
            "hi/Ri/le/iti": 90,
            "hi/Ri/le/plh": 92,
            "hi/Ri/le/tar": 94,
            "hi/Ri/le/msk": 96,
            "hi/Ri/le/fdbc": 98,
            "hi/Ri/le/fdbi": 99,
            "hi/Ri/le/fdbm": 97,
            # Ratio = 90/10 ('High), HPL = right, Target location = right
            "hi/Ri/ri/iti": 100,
            "hi/Ri/ri/plh": 102,
            "hi/Ri/ri/tar": 104,
            "hi/Ri/ri/msk": 106,
            "hi/Ri/ri/fdbc": 108,
            "hi/Ri/ri/fdbi": 109,
            "hi/Ri/ri/fdbm": 107,

            # Ratio = 80/20 ('Med-High), HPL = left, Target location = left
            "mh/Le/le/iti": 110,
            "mh/Le/le/plh": 112,
            "mh/Le/le/tar": 114,
            "mh/Le/le/msk": 116,
            "mh/Le/le/fdbc": 118,
            "mh/Le/le/fdbi": 119,
            "mh/Le/le/fdbm": 117,
            # Ratio = 80/20 ('Med-High), HPL = left, Target location = right
            "mh/Le/ri/iti": 120,
            "mh/Le/ri/plh": 122,
            "mh/Le/ri/tar": 124,
            "mh/Le/ri/msk": 126,
            "mh/Le/ri/fdbc": 128,
            "mh/Le/ri/fdbi": 129,
            "mh/Le/ri/fdbm": 127,
            # Ratio = 80/20 ('Med-High), HPL = right, Target location = left
            "mh/Ri/le/iti": 130,
            "mh/Ri/le/plh": 132,
            "mh/Ri/le/tar": 134,
            "mh/Ri/le/msk": 136,
            "mh/Ri/le/fdbc": 138,
            "mh/Ri/le/fdbi": 139,
            "mh/Ri/le/fdbm": 137,
            # Ratio = 80/20 ('Med-High), HPL = right, Target location = right
            "mh/Ri/ri/iti": 140,
            "mh/Ri/ri/plh": 142,
            "mh/Ri/ri/tar": 144,
            "mh/Ri/ri/msk": 146,
            "mh/Ri/ri/fdbc": 148,
            "mh/Ri/ri/fdbi": 149,
            "mh/Ri/ri/fdbm": 147,

             # Ratio = 70/30 ('Med-Low), HPL = left, Target location = left
            "ml/Le/le/iti": 150,
            "ml/Le/le/plh": 152,
            "ml/Le/le/tar": 154,
            "ml/Le/le/msk": 156,
            "ml/Le/le/fdbc": 158,
            "ml/Le/le/fdbi": 159,
            "ml/Le/le/fdbm": 157,
            # Ratio = 70/30 ('Med-Low), HPL = left, Target location = right
            "ml/Le/ri/iti": 160,
            "ml/Le/ri/plh": 162,
            "ml/Le/ri/tar": 164,
            "ml/Le/ri/msk": 166,
            "ml/Le/ri/fdbc": 168,
            "ml/Le/ri/fdbi": 169,
            "ml/Le/ri/fdbm": 167,
            # Ratio = 70/30 ('Med-Low), HPL = right, Target location = left
            "ml/Ri/le/iti": 170,
            "ml/Ri/le/plh": 172,
            "ml/Ri/le/tar": 174,
            "ml/Ri/le/msk": 176,
            "ml/Ri/le/fdbc": 178,
            "ml/Ri/le/fdbi": 179,
            "ml/Ri/le/fdbm": 177,
            # Ratio = 70/30 ('Med-Low), HPL = right, Target location = right
            "ml/Ri/ri/iti": 180,
            "ml/Ri/ri/plh": 182,
            "ml/Ri/ri/tar": 184,
            "ml/Ri/ri/msk": 186,
            "ml/Ri/ri/fdbc": 188,
            "ml/Ri/ri/fdbi": 189,
            "ml/Ri/ri/fdbm": 187,

             # Ratio = 60/40 ('Low), HPL = left, Target location = left
            "lo/Le/le/iti": 190,
            "lo/Le/le/plh": 192,
            "lo/Le/le/tar": 194,
            "lo/Le/le/msk": 196,
            "lo/Le/le/fdbc": 198,
            "lo/Le/le/fdbi": 199,
            "lo/Le/le/fdbm": 197,
            # Ratio = 60/40 ('Low), HPL = left, Target location = right
            "lo/Le/ri/iti": 200,
            "lo/Le/ri/plh": 202,
            "lo/Le/ri/tar": 204,
            "lo/Le/ri/msk": 206,
            "lo/Le/ri/fdbc": 208,
            "lo/Le/ri/fdbi": 209,
            "lo/Le/ri/fdbm": 207,
            # Ratio = 60/40 ('Low), HPL = right, Target location = left
            "lo/Ri/le/iti": 210,
            "lo/Ri/le/plh": 212,
            "lo/Ri/le/tar": 214,
            "lo/Ri/le/msk": 216,
            "lo/Ri/le/fdbc": 218,
            "lo/Ri/le/fdbi": 219,
            "lo/Ri/le/fdbm": 217,
            # Ratio = 60/40 ('Low), HPL = right, Target location = right
            "lo/Ri/ri/iti": 220,
            "lo/Ri/ri/plh": 222,
            "lo/Ri/ri/tar": 224,
            "lo/Ri/ri/msk": 226,
            "lo/Ri/ri/fdbc": 228,
            "lo/Ri/ri/fdbi": 229,
            "lo/Ri/ri/fdbm": 227
        }

    ### find bad epochs based on eye tracking data -----------------------------------------------------------------
    # load trigger mapping
    trig_map = pl.read_csv('eye_tracker_trigger_map.csv')
    def convert_triggers(trig_map, event):
        trig = trig_map.filter(pl.col('event') == event
                            ).select(pl.col('trigger')
                                        ).to_numpy()[0, 0]
        return trig

    # convert eyetracker event codes
    eye_events = eye.select(pl.col('event')).to_series().to_numpy()
    eye_trigs = [convert_triggers(trig_map, i) for i in eye_events[~np.isnan(eye_events)]]
    eye = eye.with_columns(pl.Series(name="triggers", values=eye_trigs)) 

    # create mne epochs object from raw eye data
    eye_data = np.swapaxes(eye.select(['L_X', 'L_Y', 'R_X', 'R_Y', 'triggers']).to_numpy(), 0, 1)
    eye_times = eye.select(['time_gd'])
    eye_ch_names = ['L_X', 'L_Y', 'R_X', 'R_Y','Status']
    eye_sfreq = 600 
    eye_ch_types = ['eyegaze', 'eyegaze', 'eyegaze', 'eyegaze', 'stim']
    eye_info = mne.create_info(ch_names=eye_ch_names, sfreq=eye_sfreq, ch_types=eye_ch_types)
    raw_eye = mne.io.RawArray(eye_data, eye_info)

    # find eye events
    eye_events = mne.find_events(raw_eye, 
        initial_event=True, 
        consecutive=True, 
        output='onset', 
        shortest_event=1)
    for key, val in event_dict.items():
        ev_mask = eye_events[:, 2] == val
        count = ev_mask.sum()
        print(f'{key}: {count}')

    # create eye epochs
    eye_epochs = mne.Epochs(
        raw=raw_eye,
        events=eye_events,
        event_id=event_dict,
        tmin=0,
        tmax=1.3,
        baseline=None,
        preload=True, 
        reject=None, 
        flat=None,
        reject_by_annotation=False,
        on_missing='warn')
    eye_epochs_plh = eye_epochs['plh']
    
    # extract eye data and find missing responses
    eye_data_plh = eye_epochs_plh.get_data(picks='eyegaze')/2 # decrease resolution by half to account for PPix 
    missing_eye_data = np.isnan(eye_data_plh)
    # cycle through each epoch and channel to find consecutive (> 2) missing data points; 
    # identify the index of the start and end of these missing periods;
    # then add another 15 samples (25 ms) either side to account for artifacts in the 
    # eye position data. (s_per_sample = 1/raw_eye.info['sfreq'] # seconds per time sample; s_per_sample*15 = 0.025)
    for e, epoch in enumerate(missing_eye_data): # cycle through epochs
        for c, chan in enumerate(epoch): # cycle through channels in each epoch
            onsets = [] # create empy lists to store onsets and offsets of missing data (i.e., blinks)
            offsets = []
            for i, j in enumerate(chan):
                # if not a missing data point, continue    
                if not j: 
                     continue
                else:
                    # if the first data point is missing, it can only be an onset, so only check for subsequent missing data
                    if i == 0: 
                        if chan[i+1]:
                            onsets.append(i)
                    # if the last data point is missing, it can only be an offset, so only check for antecedent missing data       
                    elif i == (len(chan) - 1): 
                        if chan[i-1]:
                             offsets.append(i)
                    # for all data points that aren't the first or last, check for onsets and offsets of blinks
                    else: 
                        if (not chan[i-1]) & (chan[i+1]):
                            onsets.append(i)
                        elif (chan[i-1]) & (not chan[i+1]):
                            offsets.append(i)
                        elif (chan[i-1]) & (chan[i+1]):
                            continue
            # only keep sequences of > 2 consecutive missing data points
            keeps = (np.array(onsets) - np.array(offsets)) < -2 
            # set the onset to 15 samples prior to actual onset
            onsets = np.array(onsets)[keeps]-15 
            # set the offset to 15 samples post the actual offset
            offsets = np.array(offsets)[keeps]+15 
            # if the onset is negative, set it to 0
            if np.any(np.array(onsets) < 0): 
                np.put_along_axis(onsets, np.where(onsets < 0)[0], 0, axis=0)
            # if the offset is greater than n_samples, set it to = n_samples
            if np.any(np.array(offsets) > (len(chan) - 1)): 
                np.put_along_axis(offsets, np.where(offsets > (len(chan) - 1))[0], (len(chan) - 1), axis=0)
            # update missing data array with new values
            for k in range(len(onsets)): 
                missing_eye_data[e, c, onsets[k]:offsets[k]] = True

    # now exclude the missing data 
    x_min, x_max = (1920/2)/2 - 25.5, (1920/2)/2 + 25.5 # half the normal resolution for PPixx, half that again to find centre, +- half of ROI to get 2cm either side fixation

    eye_dict = dict()
    for i, _ in enumerate(eye_data_plh):

        # combine missing data for both eyes
        nan_idxs = np.any([missing_eye_data[i, 0], missing_eye_data[i, 2]], axis=0)  
        eye_data = eye_data_plh[i, :, :][:, ~nan_idxs]
        
        # exclude trial if there isn't enough eye data 
        if eye_data.shape[-1] < 0.50*eye_data_plh.shape[-1]:
            exclude_samples = 1
        else: 
            exclude_samples = 0 

        if exclude_samples == 1:
            exclude_dev = 0
        else: 
            # average over left and right eyes and smooth data
            eye_data = np.stack([np.mean([eye_data[0], eye_data[2]], axis=0),
                                np.mean([eye_data[1], eye_data[3]], axis=0)])
            eye_data[1, :] = 1080/2 - eye_data[1, :] # invert y-axis coordinates
            eye_data = savgol_filter(eye_data, window_length=10, polyorder=1)

            # find horizontal fixations and saccades
            grad = np.gradient(eye_data[0]) 
            fix_idxs = (grad > -1) & (grad < 1)
            eye_data_fix = np.full(len(eye_data[0]), np.nan)
            eye_data_fix[fix_idxs] = eye_data[0][fix_idxs]
            dev_x = np.any([(eye_data_fix > x_max), (eye_data_fix < x_min)], axis=0)

            # exclude trials that in which the deviation exceeds 50 ms, ignoring first 50 ms
            counts = []
            count = 0
            # run on x coordinate data
            for j, k in enumerate(dev_x):
                if k: # if deviation sample, add 1 count; if last sample, add total to counts list
                    count += 1
                    if j == len(dev_x) - 1:
                        counts.append(count)
                else: 
                    if count > 0: # if there has been deviation samples previously, add total to counts list
                        counts.append(count)
                    count = 0 # reset count total
            
            if np.any(np.array(counts) > 30): # if any deviation periods greater than 50 ms, exclude trial
                exclude_dev = 1
            else: 
                exclude_dev = 0

            # plot fixations
            # fig, ax = plt.subplots()
            # ax.hlines(y=x_min, xmin=0, xmax=600, linestyles='--', color='grey')
            # ax.hlines(y=x_min + 51, xmin=0, xmax=600, linestyles='--', color='black')
            # ax.hlines(y=x_max, xmin=0, xmax=600, linestyles='--', color='grey')
            # ax.plot(eye_data[0], label = 'saccade')
            # ax.plot(eye_data_fix, label = 'fixation')
            # plt.title(f'exclude_dev: {exclude_dev}, exclude_samples: {exclude_samples}')
            # plt.legend()

        eye_dict[f'Trial_{i}'] = {
            'exclude_samples': exclude_samples,
            'exclude_dev': exclude_dev,
            #'eye': eye,
            'data_all': eye_data_plh[i],
            'data_eye': eye_data
        }

        # plot eye data
        # fig, ax = plt.subplots()
        # ax.hlines(y=x_min, xmin=0, xmax=600, linestyles='--', color='grey', label='X')
        # ax.hlines(y=x_min + 51, xmin=0, xmax=600, linestyles='--', color='black')
        # ax.hlines(y=x_max, xmin=0, xmax=600, linestyles='--', color='grey')
        # ax.hlines(y=y_min, xmin=0, xmax=600, linestyles='-.', color='grey', label='Y')
        # ax.hlines(y=y_min + 51, xmin=0, xmax=600, linestyles='-.', color='black')
        # #ax.hlines(y=y_max, xmin=0, xmax=600, linestyles='-.', color='grey')
        # ax.plot(eye_dict['Trial_1']['data_all'][0], color='red', label = 'right')
        # ax.plot(eye_dict['Trial_1']['data_all'][2], color='orange', label = 'left')
        # ax.plot(eye_dict['Trial_1']['data_eye'][0], color = 'blue', label = 'cleaned')
        # ax.plot(eye_dict['Trial_1']['data_all'][1], color='red')
        # ax.plot(eye_dict['Trial_1']['data_all'][3], color='orange')
        # ax.plot(eye_dict['Trial_1']['data_eye'][1], color = 'blue')
        # plt.legend()



    # find all bad eye epochs
    bad_eye_epochs = []
    n_missing_data = 0
    n_dev = 0
    for i, trial in enumerate(eye_dict.keys()):
        if eye_dict[trial]['exclude_samples'] == 1:
            n_missing_data += 1
            bad_eye_epochs.append(i)
        if eye_dict[trial]['exclude_dev'] == 1:
            n_dev += 1
            bad_eye_epochs.append(i)
    print(f"# Missing data trials: {n_missing_data}")
    print(f'# Deviation trial: {n_dev}')
    print(f"# Excluded trials: {len(bad_eye_epochs)}")

    # save number of bad eye epochs
    eye_epoch_dict = {
        "subID": subID,
        "#_exclude": int(len(bad_eye_epochs)),
        "# missing": int(n_missing_data)
    }

    df = pl.from_dict(eye_epoch_dict)
    if not os.path.exists(SAVEPATH + f'eye_epochs.txt'):
        df.write_csv(SAVEPATH + f'eye_epochs.txt')
    else:
        df_ld = pl.read_csv(SAVEPATH + f'eye_epochs.txt')
        df_ld.vstack(df, in_place=True)
        df_ld.write_csv(SAVEPATH + f'eye_epochs.txt')

    # save indices of bad eye epochs
    if not os.path.exists(SAVEPATH + subID):
        os.mkdir(SAVEPATH + subID)  
    with open(SAVEPATH + subID + f"/{subID}_bad_eye_epochs.txt", "w") as output:
        output.write(str(bad_eye_epochs))


### Check eye exclusion stats
eye_stats = pl.read_csv(SAVEPATH + 'eye_epochs.txt') 
eye_stats = eye_stats.with_columns((pl.col("#_exclude") - pl.col("# missing")).alias("# dev"),
                                   (pl.col("#_exclude")/960*100).alias("%_exclude"))
