#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    morphagene_onset.py -w <inputwavfile> -o <outputfile> -s <splicecount>
    
Use the Superflux onset detection algorithm with backtracking to generate 
    splice locations.
Use these splice locations with a converted WAV (to 32-bit float / 48000Hz) 
    to make Morphagene reels.
This method typically generates splices on each percussion hit of a sample,
    so be careful to choose an appropriate length sample or quickly exceed the
    limitations of the Morphagene [300 splices] using [splicecount].
Uses wavfile.py by X-Raym
"""
import librosa
import sys, getopt, os
import numpy as np
from wavfile import read, write
    
def test_normalized(array):
    '''
    Determine if an array is entirely -1 < array[i,j] < 1, to see if array is
        normalized
    '''
    return (array > -1).all() and (array < 1).all()

def norm_to_32float(array):
    '''
    Convert a variety of audio types to float32 while normalizing if needed
    '''
    if array.dtype == 'int16': 
        bits=16
        normfactor = 2 ** (bits-1)
        data = np.float32(array) * 1.0 / normfactor
        
    if array.dtype == 'int32': 
        bits=32
        normfactor = 2 ** (bits-1)
        data = np.float32(array) * 1.0 / normfactor
        
    if array.dtype == 'float32': 
        if test_normalized(array):
            print('No normalization needed')
            data = np.float32(array) # nothing needed
        else:
            bits=32
            normfactor = 2 ** (bits-1)
            data = np.float32(array) * 1.0 / normfactor

    if array.dtype == 'float64': 
        bits=64
        normfactor = 2 ** (bits-1)
        data = np.float32(array) * 1.0 / normfactor
        
    elif array.dtype == 'uint8':
        if isinstance(data[0], (int, np.uint8)):
            bits=8
            # handle uint8 data by shifting to center at 0
            normfactor = 2 ** (bits-1)
            data = (np.float32(array) * 1.0 / normfactor) -\
                            ((normfactor)/(normfactor-1))
    return data

def onset_splice_superflux(audiofile):
    '''
    Superflux onset detection method of Boeck and Widmer [2013], modified to 
        use backtracking to get accurate splice location.
    From:
    https://librosa.github.io/librosa/auto_examples/plot_superflux.html#sphx-glr-auto-examples-plot-superflux-py
    '''
    # recommended constants directly from paper
    y, sr = librosa.load(audiofile,sr=44100)
    n_fft = 1024
    hop_length = int(librosa.time_to_samples(1./200, sr=sr))
    lag = 2 # number of frames
    n_mels = 138 # number of bins
    fmin = 27.5 # lowest frequency
    fmax = 16000. #highest frequency
    max_size = 3
    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                   hop_length=hop_length,
                                   fmin=fmin,
                                   fmax=fmax,
                                   n_mels=n_mels)
    # Onset Strength Function
    odf_sf = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max),
                                      sr=sr,
                                      hop_length=hop_length,
                                      lag=lag, max_size=max_size)

    # Onset locations in time
    onset_sf = librosa.onset.onset_detect(onset_envelope=odf_sf,
                                      sr=sr,
                                      hop_length=hop_length,
                                      units='time',
                                      backtrack=True)
    return onset_sf

def retain_n_splice_markers(onset_sf, splicecount):
    '''
    modified from @w-winter on github
    
    Take larger set of generated splice points and select [splicecount]
        number of them.
    Useful for when the automatically generated number of splices exceeds
        limits (300).
    '''
    k, m = divmod(len(onset_sf), splicecount)
    if splicecount < len(onset_sf):
        splice_markers = list(onset_sf[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(splicecount))
        splice_markers[0] = [0.0]
        return np.array([x[0] for x in splice_markers])
    else:
        print('More desired splices than available splices, defaulting to librosa output')
        return onset_sf

def main(argv):
    inputwavefile = ''
    outputfile = ''
    splicecount = []
    try:
        opts, args = getopt.getopt(argv,"hw:o:s:",["wavfile=","outputfile=","splicecount="])
    except getopt.GetoptError:
        print('Error in usage, correct format:\n'+\
            'morphagene_onset3.py -w <inputwavfile> -o <outputfile> -s <splicecount>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Morphagene reel creation using Superflux onset detection:\n'+\
                  'morphagene_onset.py -w <inputwavfile> -o <outputfile> -s <splicecount>\n'+\
                  '"-s" is useful for avoiding the 300-splice limit of the Morphagene.\n'+\
                  'If you would rather bypass this, use a number >300 here.')
            sys.exit()
        elif opt in ("-w", "--wavfile"):
            inputwavefile = arg
        elif opt in ("-o", "--outputfile"):
            outputfile = arg
        elif opt in ("-s", "--splicecount"):
            splicecount = int(arg)

    print(f'Input wave file: {inputwavefile}')
    print(f'Output Morphagene reel: {outputfile}')
    print(f'Number of selected splices: {splicecount}')

    ###########################################################################
    '''
    Write single file, with splice locations using the Superflux onset 
        detection algorithm with backtracking for optimal splice location.
    '''
    ###########################################################################
    morph_srate = 48000 # required samplerate for Morphagene
    
    # generate labels and time in seconds of splices using librosa
    librosa_sec = retain_n_splice_markers(np.unique(onset_splice_superflux(inputwavefile)), splicecount)

    # read pertinent info from audio file, convert to correct 32-bit float,
    #    exit if input wave file is broken
    try:
        sample_rate, array, _, _, _ = read(inputwavefile)
        array = norm_to_32float(array)
    except: 
        print(f'Input file {inputwavefile} is poorly formatted, exiting')
        sys.exit()
    if array.ndim == 1: # correct mono
        print('Correcting mono to stereo')
        array = np.vstack((array,array)).T

    # check if input wav has a different rate than desired Morphagene rate,
    #   and correct wav by interpolation and splice positions by a scale factor
    if sample_rate != morph_srate:
        print(f"Correcting input sample rate {sample_rate}Hz to Morphagene rate {morph_srate}Hz")
        # interpolate audio file data to match morphagene sample rate
        array = librosa.resample(array, orig_sr=sample_rate, target_sr=morph_srate)
        # convert labels in seconds to frames, adjusting for change in rate
        sc = float(morph_srate) / float(sample_rate)
        frame_labs = (librosa_sec * sample_rate * sc).astype(int)
    else:
        frame_labs = (librosa_sec * sample_rate).astype(int)
    frame_dict = [{'position': l, 'label': b'marker%i'%(i+1)} for i,l in enumerate(frame_labs)]
    
    # force 2-column array for correct formatting
    if array.shape[1] > 2:
        print('Correcting wav array format')
        array = array.T

    # warnings about morphagene limitations
    if len(frame_dict) > 300 or (array.shape[1]/morph_srate)/60. > 2.9:
        raise ValueError(f'Number of splices ({len(frame_dict)}) and/or audio'+ \
            f' length ({(array.shape[1]/morph_srate)/60.} minutes)' + \
            ' exceed Morphagene limits [300 splices / 2.9 minutes]')

    # write normalized wav file with additional cue markers from labels
    write(outputfile,morph_srate,array.astype('float32'),
          markers=frame_dict,
          normalized=True)
    print(f'Saved Morphagene reel with {len(frame_labs)} splices: {outputfile}')
    name = os.path.splitext(inputwavefile)[0]
    np.savetxt(f'{name}_splices.txt',librosa_sec,fmt='%03.6f',delimiter='\t')
    
if __name__ == "__main__":
    main(sys.argv[1:])
