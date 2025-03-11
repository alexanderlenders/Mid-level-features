#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG PREPROCESSING FOR DECODING 

This script implements the preprocessing of the EEG data for the decoding 
analysis. It downsamples the EEG data to 50hz to increase the computational 
efficiency. It epochs the data into 1380ms long epochs. 

Acknowledgement: This script is based on a preprocessing script by Vanshika
Bawa. 

Anaconda-Environment on local machine: MNE

@author: AlexanderLenders
"""
# -----------------------------------------------------------------------------
# STEP 1: Initialize variables
# -----------------------------------------------------------------------------
if __name__ == '__main__': 
    import argparse # if script is run in terminal $ python preprocessing_eeg.py -h

    # parser
    parser = argparse.ArgumentParser()
    
    # add arguments / inputs
    parser.add_argument('-s', "--sub", default=5, type=int, metavar='', 
                        help="subject number")
    parser.add_argument('-se', "--sess", default=1, type=int, metavar='', 
                        help="subject session")
    parser.add_argument('-f', "--freq", default=50,type=int, metavar='', 
                        help="downsampling frequency")
    parser.add_argument('-r', "--region", default="posterior",type=str, 
                        metavar='',
                        help="Electrodes to be included, posterior (19) or wholebrain (64)")
    parser.add_argument('-d', "--workdir", 
                        default = '/Users/AlexanderLenders/Desktop/',
                        type = str, metavar='', help="Working directory: where EEG and behavioral data is stored")
    parser.add_argument('-p', "--plots", default=True,
                        type = bool, metavar='', help="Show diagnostic plots?")
    parser.add_argument('-sf', "--sfigures", default=True, type = bool,
                        metavar='', help="Save figures? Recommended only for ICA.\
                            Does only work, if sfigures = TRUE.")
    parser.add_argument('-i','--ica', default=True, type=bool, help = "Do an ICA?")
    parser.add_argument('-it', '--input_type', default='images', type=str, metavar='', 
                        help='images or miniclips')

    args = parser.parse_args() # to get the values for the arguments
    
    # rename arguments 
    sub = args.sub
    sess = args.sess
    freq = args.freq
    select_channels = args.region            
    workDir = args.workdir
    do_ica = args.ica
    show_plots = args.plots
    save_figures = args.sfigures
    input_type = args.input_type

# -----------------------------------------------------------------------------
# STEP 2: Preprocessing (Define function preprEeg)
# -----------------------------------------------------------------------------

def preprEeg(sub, sess, freq, select_channels, do_ica, workDir, 
             show_plots, save_figures, input_type):
    """Preprocessing of the EEG data

    Input: 
    ----------
    EEG data files (BrainVision-Recorder)

    Returns:
    ----------
    Preprocessed EEG-Data splitted into training, test and validation 
    data set for the encoding analysis. Note that the decoding analysis is only
    done with the test dataset. 
    The output is therefore 3 dictionaries which contain: 
    a. EEG-Data (eeg_data, 5400 Stimuli x 64 Channels x 70 Timepoints)
    b. Stimulus Categories (img_cat, 5400 x 1) - Each stimulus has one specific ID
    c. Channel names (channels, 64 x 1 OR 19 x 1)
    d. Time (time, 70 x 1) - Downsampled timepoints of a stimulus
    In case of the validation data set there are 900 stimuli instead of 5400.

    Arguments: 
    ----------
    sub : int
          Subject number.
    sess : int
          Subject's session of the paradigm (default is 1).
    freq : int
          Downsampling frequency (default is 50).
    region : str
        The region for which the EEG data should be analyzed, defaut is 
        wholebrain.
    workDir: str 
        Working directory (folders should be in BIDS-format)
    do_ica: bool
        Do ICA on data
    show_plots: bool
        Show diagnostic plots
    save_figures: bool 
        Save diagnostic figures
    input_type: str
        Images or Miniclips

    """
    # -------------------------------------------------------------------------
    # STEP 2.1: Initialize Variables + Import Modules
    # -------------------------------------------------------------------------
    
    # Import modules
    import mne
    import numpy as np
    from scipy import io
    import pandas as pd
    import matplotlib
    import os

    print(">>> Preprocessing %dhz, sub %s, brain region: %s <<<" % 
          (freq, sub, select_channels))
    
    # Even if we are only interested in the posterior electrodes, the ICA 
    # should be done based on all the available electrodes.
    if do_ica: 
        selected_channels = select_channels # in case we only want posteriors
        select_channels = "wholebrain"
    else: 
        selected_channels = select_channels

    # Stimuli per sequence 
    img_per_seq = 10 
    
    # data directories (works only on mac/linux - change format on Windows)
    if input_type == 'images':
        workDir_input = os.path.join(workDir, 'images_data')
        file_identifier = 'unreal_subject_0' if sub < 10 else 'unreal_subject'
    elif input_type == 'miniclips':
        workDir_input = os.path.join(workDir, 'miniclip_data')
        file_identifier = 'miniclip_vis_features_00' if sub < 10 else 'miniclip_vis_features_0'


    if sub < 10: 
        behavDir = workDir_input + ('/sub-0{}'.format(sub) + 
                              '/beh/temporal_decoder_data')
        eegDir = workDir_input + ('/sub-0{}'.format(sub) + 
                              '/eeg/{}{}'.format(file_identifier,sub) +
                              '.vhdr')
    else:
        behavDir = workDir_input + ('/sub-{}'.format(sub) + 
                              '/beh/temporal_decoder_data')
        eegDir = workDir_input + ('/sub-{}'.format(sub) + 
                              '/eeg/{}{}'.format(file_identifier,sub) +
                              '.vhdr')
    
    # -------------------------------------------------------------------------
    # STEP 2.2: Load behavioral data and transform them into dataframe
    # -------------------------------------------------------------------------
    # io.loadmat imports the .mat structure as a dict
    behav = io.loadmat(behavDir) # imports .mat str as a dict 
    data_key = behav['data'] 
    behav_img = data_key[0][0]["images"] 
    # data.images contains table with information about stimuli shown, img_type etc.
    
    def behav_to_df(behav_img):
        # create pandas dataFrame from behavioral dataset
        headers = list(behav_img[:,0].dtype.names)
        df = pd.DataFrame() # creates empty data frame
        for count, head in enumerate(headers):
            print(count, head)
            dtype = object
            col = np.asarray(behav_img[head], dtype = dtype)
            df[head] = col[0]
        return df, headers

    behavior_df, headers = behav_to_df(behav_img) # run function
    # define dtypes of the variables in headers: 
    dtypes = [int, int, int, str, str, str, int, int, int, float, float]
    # create a dictionary out of these two lists:
    dtype_dict = dict(zip(headers, dtypes))
    # cast the data frame to the specified dtype
    behavior_df = behavior_df.astype(dtype_dict)
    
    img_types = ['test', 'training', 'validation']


    # -------------------------------------------------------------------------
    # STEP 2.3: Import EEG data and select channels
    # -------------------------------------------------------------------------
    
    # import EEG data
    raw = mne.io.read_raw_brainvision(eegDir, preload=True)
    
    # create a montage for plotting with electrode positions later
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    
    # sampling frequency 
    original_frequency = raw.info['sfreq']

    # selecting channels 
    ch_names = raw.ch_names # names of the 64 channels
    
    if select_channels == "posterior":
        sel_chan = "^O *|^P *" #regex
        chan_idx = mne.pick_channels_regexp(ch_names, sel_chan)
    elif select_channels == "wholebrain":
        chan_idx = mne.pick_channels(ch_names, include = []) 

    chan_idx = np.asarray(chan_idx) 
    
    # create a list (list comprehension)
    sel_chans = [ch_names[channel_idx] for channel_idx in chan_idx]
    raw.pick_channels(sel_chans) # pick only these channels

    # -------------------------------------------------------------------------
    # STEP 2.4: Annotating continous data (events)
    # -------------------------------------------------------------------------
    """ 
    Annotations are list-like object, where each element comprises three 
    pieces of information: (1) onset time (sec), (2) duration (sec), 
    (3) description.
    Events are different structures, consisting of (1) event time: 
    sample number of the onset, (2) pulse magnitudes as integer codes.
    """

    # convert annotations to events
    events, events_id = mne.events_from_annotations(raw)
    
    # to exclude events, which are not of interest (10001, 10002)
    events = events[np.logical_and(events[:,2] != 10001, events[:,2] != 10002)]
    events = events[np.logical_and(events[:,2] != 99999,events[:,2] != 10003)]

    # in case there are too many events (eyetracker) - e.g., for subject 10, you'll
    # also have to run the following commented line: 
    # events = events[1:, :]

    # create empty arrays
    timestamps = [] # this will contain the different timings
    identifiers = [] # this will contain the eeg ids (matching the behavioural ids)
    
    # -------------------------------------------------------------------------
    # STEP 2.5: Convert EEG-Triggers back into Behavioural ID's
    # -------------------------------------------------------------------------
    """
    events.shape[0]/2) -> number of rows in column 0 divided by 2
    It is divided by 2 because 2 triggers were sent to the EEG system to 
    indicate an behavioural event (since the eeg recording system could not
    deal with ids starting with "0"). 
    If the behavioural id started with 0 -> 0 was replaced with 255 in 
    an eeg-trigger.
    """
    # The for-loop reconstructs the two different ids: 

    for i in range(int(events.shape[0]/2)):
        time = events[i*2, 0] # time := sample number of the onset
        trig1 = events[i*2, 2]
        tr1 = trig1
        trig2 = events[(i*2)+1,2]
        tr2 = trig2
    # This reconstructs the behavioural ids (see above):
        if (trig1 == 255):
            trig1 = 0
        if (trig2 == 255):
            trig2 = 0
        trig = f'{trig1:02}' + f'{trig2:02}'
        if trig == 0: 
            print(i, tr1, tr2) # Why?

        timestamps.append(time)
        identifiers.append(trig)
    
    # -------------------------------------------------------------------------
    # STEP 2.6: Compare reconstructed EEG-Triggers with Behavioural ID's
    # -------------------------------------------------------------------------
    
    # create an array with the behavioural id's: 
    img_cat = np.array(behavior_df["behav_trigger"])

    # create an array with the reconstructed behavioural id's from eeg triggers:
    ident_int = np.array([int(i) for i in identifiers])
    
    # they should be identical (if everything has worked well):
    if (np.sum(img_cat == ident_int) != len(img_cat)):
        print("WARNING, the triggers do not agree with each other")
        
    # -------------------------------------------------------------------------
    # STEP 2.7: Filter for test/validation or training sequences
    # -------------------------------------------------------------------------
    # depending on the img_type: find the indices of test/val or training 
    # sequences: 
 
    # convert series into integers 
    con_identifiers = np.array(identifiers)
    con_timestamps = np.array(timestamps)
    
    # seq_idx contains only the first stimulus of a sequence: 
    seq_idx = behavior_df["img"] == 1
    
    # this array contains only the first stimulus of the sequences: 
    seq_identifiers = con_identifiers[seq_idx]
    seq_timestamps = con_timestamps[seq_idx]
    
    # this array contains only the first stimulus of the sequences:
    seq_img_cat = img_cat[seq_idx]
    
    # do the same for the different img_types (test, val, training)
    indices = {}
    img_type_cat = {}
    identifiers_type = {}
    timestamps_type = {}
    seq_identifiers_type = {}
    seq_timestamps_type = {}
    seq_img_type = {}
    
    for img in img_types: 
        index = np.array(behavior_df["img_type"] == "['{}']".format(img))
        indices[img] = index
        
        img_type_cat[img] = img_cat[index]
        identifiers_type[img] = np.array(identifiers)[index]
        timestamps_type[img] = np.array(timestamps)[index]
        seq_idx_type = behavior_df["img"][index] == 1
        
        seq_identifiers_type[img] = identifiers_type[img][seq_idx_type]
        seq_timestamps_type[img] = timestamps_type[img][seq_idx_type]
        seq_img_type[img] = img_type_cat[img][seq_idx_type]
    
    # create an array which has the shape of the events array of the MNE 
    # package: 
    seq_events = np.array(
        [seq_timestamps, 
        np.zeros(len(seq_identifiers), dtype = int),
        seq_img_cat]).T

    # In case of high-frequency noise at 50hz, do in addition to the low-pass
    # filter (see below) notch filter: 
    # if show_plots: 
        # Power line frequency in Germany 50hz
        # Plot power spectrum of the raw, unfiltered data
    #    power_frequency_raw = raw.plot_psd(tmax=np.inf, fmax=250, average=True)
        
    # We can filter the line noise: 
    freqs = (50) # define frequency which we want to attenuate
    raw = raw.notch_filter(freqs = freqs)
    
    # -------------------------------------------------------------------------
    # STEP 2.8: Epoching sequences (without baseline correction)
    # -------------------------------------------------------------------------
    # Epochs are similar to raw objects, and inside an epoch object the data
    # are stored in an array of shape (n_epochs, n_channels, n_times)
    
    # tmin = -1 (epoch starts 1 second before the event)
    epochs = mne.Epochs(raw, seq_events, tmin=-1, tmax=6, baseline=None, 
                        preload=True)
    
    # -------------------------------------------------------------------------
    # STEP 2.9: ICA
    # -------------------------------------------------------------------------
    """ 
    MNE implements three different algorithms - fastica is the default, picard
    might be a bit more robust (see MNE documentation). 
    Results of the fitting are added to an ICA object with underscores 
    - ica.mixing_matrix_ 
    - ica.unmixing_matrix
    Here the fast algorithm was used. 
    We could also set n_components = 64 (because we had 64 channels), in case we
    use less components we will do a dimensionality reduction. 
    """
    if do_ica: 
        # High-pass filtering will be applied, but reconstruction on unfiltered
        # data (1Hz, see MNE documentation for more information). 
        raw = raw.filter(l_freq = 1., h_freq = None)
        filtered_epochs = mne.Epochs(raw, seq_events, tmin=-1, tmax=6, 
                                     baseline=None, preload=True)
        
        original_epochs = epochs.copy() # make a copy
        # ICA is non deterministic, thus we have to determine a random seed = 97
        # We could also say max_iter = 800. 
        ica = mne.preprocessing.ICA(random_state=97, max_iter= 'auto',
                                    n_components=54)
        ica.fit(filtered_epochs) 
        
        # get information about the explained variance of the ICA components: 
        explained_var_ratio = ica.get_explained_variance_ratio(filtered_epochs)
        for channel_type, ratio in explained_var_ratio.items():
            print(
                f'Fraction of {channel_type} variance explained by all components: '
                f'{ratio}'
            )

        # get explained variance of the first 30 components: 
        mixing_matrix = ica.mixing_matrix_
        explained_variance = np.square(mixing_matrix).sum(axis=0) / np.square(mixing_matrix).sum()

        # create array with component index and explained variance ratio
        components = np.arange(len(explained_variance))
        exp_var_array = np.column_stack((components, explained_variance))

        # plot the time series of the independent components
        if show_plots:
            ica.plot_sources(epochs)
            # visualize the scalp field distribution of the components
            ica.plot_components(inst = epochs)
            # diagnostics of each IC 
            components = [i for i in range(0,54)]
            ica.plot_properties(epochs, components)
        
        # exclude a component by hand - determine the components for each subject
        # individually:
        ica.exclude = [49, 39, 34, 31, 28, 27, 24, 21, 20, 19, 18, 15, 12, 11, 9, 8, 4, 3, 1]
        
        # reconstruct signal without excluded component 
        ica.apply(epochs)
        
        if show_plots:
            epochs.plot(group_by='position', butterfly=True, show_scrollbars=False, 
                        n_epochs=10)
            original_epochs.plot(group_by='position', butterfly=True, 
                                show_scrollbars=False, n_epochs=10)
            # Power spectrum after ICA:
            power_after_ica = epochs.plot_psd(tmax=np.inf, fmax=200, average=True)
        
        del original_epochs # save memory
        del filtered_epochs # save memory

    del raw # we're done with raw, free up some memory

    # -------------------------------------------------------------------------
    # STEP 2.10: Downsampling
    # -------------------------------------------------------------------------
    # Dowsampling the sequence data + this automatically applies a low-pass
    # filter to the data to avoid aliasing (at the Nyquist frewuency of the desired
    # frequency, i.e. in the case of 50hz the Nyquist frequency is 25 hz, thus the
    # low-pass filter has a cut-off of 25hz). Therefore, we also do not need to 
    # apply a line noise filter.
    # https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html

    before_downsampling = epochs.copy()
    epochs.resample(freq)
    
    if show_plots: 
    # show the lowpass filter:
        before_low_pass = before_downsampling.plot_psd(tmax=np.inf, fmax=250,
                                                       average=True)
        after_low_pass = epochs.plot_psd(tmax=np.inf, average=True)
        
    del before_downsampling # to save memory
    
    # -------------------------------------------------------------------------
    # STEP 2.11: Select channels 
    # -------------------------------------------------------------------------
    # We have to select the channels again in case we did ICA
    if do_ica: 
        if selected_channels == 'posterior': 
            sel_chan_regex = "^O *|^P *"
            chan_idx = mne.pick_channels_regexp(epochs.info["ch_names"], 
                                                sel_chan_regex)
            chan_idx = np.asarray(chan_idx)
            sel_chans = [epochs.info["ch_names"][c] for c in chan_idx]
            epochs.pick_channels(sel_chans)
    
    # ------------------------------------------------------------------------
    # We have to do the last steps for each img_type. To do that, we have to
    # filter the epochs. 
    for img_type in img_types: 

        events_in_epoch = epochs.events
        event_index = seq_img_type[img_type]
        mask = np.isin(events_in_epoch[:, 2], event_index)
        epochs_cond = epochs[mask]
        
        # The time vector is the timing of the EEG-samples (1400) per sequence,
        # that means starting from -1s to 6s (in seconds)
        channel_names = epochs_cond.ch_names
        seq_time = epochs_cond.times
        seq_data = epochs_cond.get_data()
        
        n_img_epochs = seq_data.shape[0] * img_per_seq
        
        # -------------------------------------------------------------------------
        # STEP 2.12: Define single stimuli as events (not sequences as above)
        # -------------------------------------------------------------------------

        # Redefining the length of the epochs
        time_idx = np.where(seq_time == 0)[0] # first stimulus
        tmin = int(time_idx - 400 / (original_frequency / freq)) # epoch_start(ms) / 1000 / resampling_freq(hz))
        tmax = int(time_idx + 1000 / (original_frequency / freq)) # epoch_end(ms) / 1000 / resampling_freq(hz))
        img_time = seq_time[tmin:tmax]
    
        # Epochs * channels * time: Create an empty array with 5400 x 64 x 1400
        eeg_data = np.empty((n_img_epochs, seq_data.shape[1], len(img_time)))
        eeg_data[:] = np.nan # NA values in every entry
        trl_count = 0
        
        # Apply the changed length of the epochs + baseline correction:
        for s in range(seq_data.shape[0]): # for every sequence (out of 540)
            time_idx = np.where(seq_time == 0)[0] 
            
            for i in range(img_per_seq): # for every stimulus in the sequence (out of 10)
            
                tmin = int(time_idx - 400 / (1000 / freq)) # epoch_start(ms) / 1000 / resampling_freq(hz))
                tmax = int(time_idx + 1000 / (1000 / freq)) # epoch_end(ms) / 1000 / resampling_freq(hz))
                
                img_data = seq_data[s,:,tmin:tmax]
    
                # Baseline correcting the data (absolute baseline)
                base = np.mean(img_data[:,int(300 / (1000 / freq)):int(400 / (1000 / freq))], 
                               axis=1) # 100ms baseline ### should probably be (1000/freq)
                base = np.reshape(base, (len(base), 1))
                img_data = img_data - base # baseline correction
                eeg_data[trl_count,:,:] = img_data
                
                time_idx += int(400 / (1000 / freq)) # +400ms (new onset of eeg data)
                trl_count += 1
    
    
        # -------------------------------------------------------------------------
        # STEP 2.13: Diagnostic plots 
        # -------------------------------------------------------------------------
        # Those diagnostic plots show the ERPs and are a sanity check before the 
        # decoding analysis is conducted. If the preprocessing has worked, we should
        # see components which are characteristic for early visual processes. 

        if show_plots: 
            
            # Let's look at the average epoch for each channel: 
            avg_epochs = np.mean(eeg_data, axis = 0)
            average_epoch, axes = matplotlib.pyplot.subplots(nrows = 8, ncols = 8, 
                                                   figsize =(30,30))
            matplotlib.pyplot.subplots_adjust(wspace=1, hspace=1, left=0.1, 
                                              right=0.9, top=0.9, bottom=0.1)
            for i, ax in enumerate(axes.flatten()): 
                ax.plot(avg_epochs[i, :])
                ax.set_title('Channel {}'.format(i+1), fontsize = 8)
                ax.set_xlabel('Time', fontsize = 5)
                ax.set_ylabel('Amplitude', fontsize = 5)
                ax.set_xlim(0, 70)
                ax.set_xticks([0, 70])
                ax.set_xticklabels([-0.4, 1])
        
            matplotlib.pyplot.show()
            
           #---------------------------------------------------------------------
           
            # Create a x-axis in ms
            ticks = list(range(0,70))
            ticks = ticks[0:70:20]
            labels = [round(item, 2) for item in img_time[ticks]]
            
            # Let's look at C1 component, which is largest at posterior midline
            # electrode sides. This component is generated by V1 and peaks at around
            # 80-100 ms. 
            # Select the channels to plot
            channels = [24, 29, 61] #Pz #POz #Oz
            
            # Filter the avg_epochs array to include only the selected channels
            posterior_midline = avg_epochs[channels, :]
            
            # Create figure for each channel and subplots
            c1_each, axes = matplotlib.pyplot.subplots(nrows=3, ncols=1, figsize=(30, 30))
            matplotlib.pyplot.subplots_adjust(wspace=1, hspace=1, left=0.1, right=0.9, 
                                              top=0.9, bottom=0.1)
            
            # Loop over the subplots and plot the data for each channel
            for i, ax in enumerate(axes.flatten()):
                ax.plot(posterior_midline[i, :])
                ax.set_title('Channel {}'.format(channels[i] + 1), fontsize=8)
                ax.set_xlabel('Time', fontsize=5)
                ax.set_ylabel('Amplitude', fontsize=5)
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels)
                
            # Show the figure
            matplotlib.pyplot.show()
        
            #  --------------------------------------------------------------------
            # Let's look at P1 component, which is largest at leateral occipital 
            # electrodes and onsets 60-90 ms. 
           
            # Select the channels to plot
            channels = [28, 29, 30, 59, 63, 60, 62] 
           
            # Filter the avg_epochs array to include only the selected channels
            lateral_occipital = avg_epochs[channels, :]
           
            # Create figure for each channel and subplots
            p1_each, axes = matplotlib.pyplot.subplots(nrows = 7, ncols =1, figsize=(30, 30))
            matplotlib.pyplot.subplots_adjust(wspace=1, hspace=1, left=0.1, right=0.9, 
                                             top=0.9, bottom=0.1)
           
            # Loop over the subplots and plot the data for each channel
            for i, ax in enumerate(axes.flatten()):
                ax.plot(lateral_occipital[i, :])
                ax.set_title('Channel {}'.format(channels[i] + 1), fontsize=8)
                ax.set_xlabel('Time', fontsize=5)
                ax.set_ylabel('Amplitude', fontsize=5)
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels)
               
            # Show the figure
            matplotlib.pyplot.show()

        # -------------------------------------------------------------------------
        # STEP 2.14: Save sequences + behavior_df
        # -------------------------------------------------------------------------
        # Putting the data into a dictionary 
        if input_type == 'images':
            image_data = {
                    "eeg_data": eeg_data,
                    "img_cat": img_type_cat[img_type],
                    "channels": channel_names,
                    "time": img_time
            }
        elif input_type == 'miniclips':
            video_data = {
                    "eeg_data": eeg_data,
                    "img_cat": img_type_cat[img_type],
                    "channels": channel_names,
                    "time": img_time
            }
        
        if sub < 10: 
            # Saving the eeg data (works only on Mac OS/Linux)
            if do_ica:
                saveDir =  workDir_input + ('/sub-0{}'.format(sub) + '/eeg/preprocessing/ica/'
                                      + img_type + '/' + selected_channels)
            else:
                 saveDir =  workDir_input + ('/sub-0{}'.format(sub) + '/eeg/preprocessing/no_ica/' +
                                       img_type + '/' + selected_channels)
            fileDir = ('sub-0{}'.format(sub) + '_seq_' + img_type + '_' + str(freq) + 'hz_' 
                       + selected_channels)
            
            # Creating the directory if not existing
            if not os.path.isdir(os.path.normpath(os.path.join(workDir_input, saveDir))):
                os.makedirs(os.path.normpath(os.path.join(workDir_input, saveDir)))
        
            if input_type == 'images':
                np.save(os.path.join(saveDir, fileDir), image_data)
            elif input_type == 'miniclips':
                np.save(os.path.join(saveDir, fileDir), video_data)
            
            if do_ica:
                cfileDir = ('sub-0{}'.format(sub) + 'explained_variance_ICA')
                np.save(os.path.join(saveDir, cfileDir), exp_var_array)
        else: 
            # Saving the eeg data (works only on Mac OS/Linux)
            if do_ica:
                saveDir =  workDir_input + ('/sub-{}'.format(sub) + '/eeg/preprocessing/ica/'
                                      + img_type + '/' + selected_channels)
            else:
                 saveDir =  workDir_input + ('/sub-{}'.format(sub) + '/eeg/preprocessing/no_ica/' +
                                       img_type + '/' + selected_channels)
            fileDir = ('sub-{}'.format(sub) + '_seq_' + img_type + '_' + str(freq) + 'hz_' 
                       + selected_channels)
            
            # Creating the directory if not existing
            if not os.path.isdir(os.path.normpath(os.path.join(workDir_input, saveDir))):
                os.makedirs(os.path.normpath(os.path.join(workDir_input, saveDir)))
        
            if input_type == 'images':
                np.save(os.path.join(saveDir, fileDir), image_data)
            elif input_type == 'miniclips':
                np.save(os.path.join(saveDir, fileDir), video_data)
            
            if do_ica:
                cfileDir = ('sub-{}'.format(sub) + 'explained_variance_ICA')
                np.save(os.path.join(saveDir, cfileDir), exp_var_array)
    
        if sub < 10: 
            # Saving the behavior df
            behav_dir = workDir_input + ('/sub-0{}'.format(sub) + '/beh/' + 'behavior_df.csv')
            behavior_df.to_csv(behav_dir)
        else: 
            behav_dir = workDir_input + ('/sub-{}'.format(sub) + '/beh/' + 'behavior_df.csv')
            behavior_df.to_csv(behav_dir)
        
    
# -----------------------------------------------------------------------------
# STEP 3: Run function
# -----------------------------------------------------------------------------
preprEeg(sub, sess, freq, select_channels, do_ica, workDir, show_plots, 
        save_figures)


