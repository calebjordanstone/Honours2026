import polars as pl
import mne
import numpy as np 
import os
from pathlib import Path
from eeg_fasterAlgorithm import *
%matplotlib qt

## set data paths
DATAPATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/honours_projects/Honours2026/data/sourcedata" 
SAVEPATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/honours_projects/Honours2026/data/derivatives/preprocessed/"
srcDataEEG = sorted(Path(DATAPATH).glob('**/*.bdf'))

# extract some BIDS info
path = srcDataEEG[40]
subID, task, modality = path.stem.split('_')

# load data
raw = mne.io.read_raw_bdf(
    path,
    preload=True)

# set montage
raw.set_montage(
    'biosemi64', 
    on_missing='warn')

# add EOG channels
info = mne.create_info( 
    ch_names=['hEOG','vEOG'], 
    sfreq=raw.info['sfreq'], 
    ch_types='eog'
    )
RH, LH, LV, UV = mne.pick_channels(
    raw.ch_names, 
    ['EXG5', 'EXG6', 'EXG7', 'EXG8']
    )
hEOG = raw[LH][0] - raw[RH][0]
vEOG = raw[UV][0] - raw[LV][0]
newEOG = mne.io.RawArray(
    np.concatenate([hEOG, vEOG]), 
    info=info)
raw.add_channels(
    add_list=[newEOG], 
    force_update_info=True)
raw.drop_channels(
    ch_names=['EXG1', 'EXG2', 'EXG3', 'EXG4', 
            'EXG5', 'EXG6', 'EXG7', 'EXG8'])

# find events
eeg_events = mne.find_events(raw, 
    initial_event=True, 
    consecutive=True, 
    output='onset', 
    shortest_event=1)

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

event_dict_plh = {
        # Ratio = 90/10 ('High), HPL = left, Target location = left
        "hi/Le/le/plh": 72,
        # Ratio = 90/10 ('High), HPL = left, Target location = right
        "hi/Le/ri/plh": 82,
        # Ratio = 90/10 ('High), HPL = right, Target location = left,
        "hi/Ri/le/plh": 92,
        # Ratio = 90/10 ('High), HPL = right, Target location = right
        "hi/Ri/ri/plh": 102,
        # Ratio = 80/20 ('Med-High), HPL = left, Target location = left
        "mh/Le/le/plh": 112,
        # Ratio = 80/20 ('Med-High), HPL = left, Target location = right
        "mh/Le/ri/plh": 122,
        # Ratio = 80/20 ('Med-High), HPL = right, Target location = left
        "mh/Ri/le/plh": 132,
        # Ratio = 80/20 ('Med-High), HPL = right, Target location = right
        "mh/Ri/ri/plh": 142,
        # Ratio = 70/30 ('Med-Low), HPL = left, Target location = left
        "ml/Le/le/plh": 152,
        # Ratio = 70/30 ('Med-Low), HPL = left, Target location = right
        "ml/Le/ri/plh": 162,
        # Ratio = 70/30 ('Med-Low), HPL = right, Target location = left
        "ml/Ri/le/plh": 172,
        # Ratio = 70/30 ('Med-Low), HPL = right, Target location = right
        "ml/Ri/ri/plh": 182,
        # Ratio = 60/40 ('Low), HPL = left, Target location = left
        "lo/Le/le/plh": 192,
        # Ratio = 60/40 ('Low), HPL = left, Target location = right
        "lo/Le/ri/plh": 202,
        # Ratio = 60/40 ('Low), HPL = right, Target location = left
        "lo/Ri/le/plh": 212,
        # Ratio = 60/40 ('Low), HPL = right, Target location = right
        "lo/Ri/ri/plh": 222
    }

# plot events
# mne.viz.plot_events(eeg_events, 
#                     event_id=event_dict,
#                     on_missing='warn',
#                     sfreq=raw.info['sfreq'])

# check event counts
for key, val in event_dict.items():
    ev_mask = eeg_events[:, 2] == val
    count = ev_mask.sum()
    print(f'{key}: {count}')

# mark breaks in data collection
break_annots = mne.preprocessing.annotate_break(
    raw=raw,
    events=eeg_events,
    min_break_duration=5,  # consider segments of at least 5 s duration
    t_start_after_previous=2,  # start annotation 3 s after end of previous one
    t_stop_before_next=2,  # stop annotation 3 s before beginning of next one
) 
raw.set_annotations(break_annots)

# plot raw data
#raw.plot()

## Start actual pre-processing ---------------------------------------------------------

# filter EEG data
filt_h = raw.copy().filter(
    l_freq=None, 
    h_freq=99, 
    picks='eeg')

# identify bad chans
filt_h.info['bads'] += faster_bad_channels(filt_h, 
                                        use_metrics=['correlation',
                                                        'variance',
                                                        'hurst'])

# check filtered data to make sure no bad channels were missed and mark bad sections
filt_h.plot()

# handle bad channels
intrp = filt_h.copy().interpolate_bads()   

# re-reference to average 
avRef = intrp.set_eeg_reference()

# create two copies of data, one for ICA and one for continued preprocessing
ica_data = avRef.copy().filter(l_freq=1.0, h_freq=None, picks='eeg') # use unfiltered data to apply higher high-pass filter
filt_l = avRef.copy().filter(l_freq=0.1, h_freq=None, picks='eeg')

# run ICA
ICA = mne.preprocessing.ICA() 
ICA.fit(ica_data, decim=8) # speed up processing

# check ICA components on filtered data
# automatically find the ICs that best match the EOG signal
eog_indices, eog_scores = ICA.find_bads_eog(filt_l)
print(f'EOG indicies: {eog_indices}')
ICA.exclude = eog_indices

# apply ICA 
ICA.apply(filt_l)
filt_l.plot()

# create epochs
epochs_plh = mne.Epochs(
    raw=filt_l,
    events=eeg_events,
    event_id=event_dict_plh,
    tmin=-0.8,
    tmax=3,
    baseline=[-0.2, 0],
    preload=True, 
    reject=None, 
    flat=None,
    reject_by_annotation=False,
    on_missing='warn')

# identify bad EEG epochs
bads_epochs = faster_bad_epochs(epochs_plh)

# save epochs
epoch_dict = {}
epoch_dict['subID'] = subID
for key, val in event_dict_plh.items():
    ev_mask = eeg_events[:, 2] == val
    count = ev_mask.sum()
    epoch_dict[f'{key.replace("/", "_")[0:-4]}'] = int(count) 
epoch_dict['n_bd_chns'] = len(filt_h.info['bads'])

# save number of epochs per condition and number of bad channels
df = pl.from_dict(epoch_dict)
if not os.path.exists(SAVEPATH + f'epochs_plh.txt'):
    df.write_csv(SAVEPATH + f'epochs_plh.txt')
else:
    df_ld = pl.read_csv(SAVEPATH + f'epochs_plh.txt', schema_overrides={'av_trls':pl.String})
    df_ld.vstack(df, in_place=True)
    df_ld.write_csv(SAVEPATH + f'epochs_plh.txt')
# save plh epoch files
if not os.path.exists(SAVEPATH + subID):
    os.mkdir(SAVEPATH + subID)
FILENAME = f'/{subID}_{task}_desc-epochs_{modality}.fif'
epochs_plh.save(SAVEPATH + subID + FILENAME)
# save indices of bad eeg epochs
with open(SAVEPATH + subID + f"/{subID}_bad_eeg_epochs.txt", "w") as output:
    output.write(str(bads_epochs))

### Check bad epoch stats
import re
bad_epochs_paths = sorted(Path(SAVEPATH).glob(f'**/*_bad_eeg_epochs.txt')) 
# get bad eeg epochs
bads = []
for path in bad_epochs_paths: 
    with open(path, 'r') as file:
        eeg_bads_str = file.readlines()
    eeg_bads = re.findall(r"-?\d*\.?\d+", eeg_bads_str[0])
    eeg_bads = list(map(int, eeg_bads)) # convert to list of ints
    bads.append(len(eeg_bads))
