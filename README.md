# morphagene_onset
Use Superflux onset detection with backtracking to generate splice locations for use with the Make Noise Morphagene.

Requires librosa and wavfile.py (from X-Raym, https://github.com/X-Raym/wavfile.py/blob/master/wavfile.py).

# Example
Extract and place 64 splices (if >64 splices calculated) in selected wav file, and save as `spliced_reel` 
```
python morphagene_onset.py -w "path/to/morphagene/reel/' -o 'spliced_reel' -s 64
```
