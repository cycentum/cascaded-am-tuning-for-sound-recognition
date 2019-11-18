# cascaded-am-tuning-for-sound-recognition
Codes for "Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition" by Takuya Koumura, Hiroki Terashima, and Shigeto Furukawa.
## Reference
Koumura T, Terashima H, Furukawa S (2019) Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition. J Neurosci 39(28):5517–5533. DOI: https://doi.org/10.1523/JNEUROSCI.2914-18.2019
## Dependencies
- Python 3
- Chainer https://chainer.org/
- soundfile https://pypi.org/project/SoundFile/
## Datasets
- Datasets at figshare https://doi.org/10.6084/m9.figshare.7914611
  - Unzip the 7z files, put them in a single directory named "cascaded-am-tuning-for-sound-recognition", and put this in the same directory as the codes.
  - The directory tree should look like this:
```
cascaded-am-tuning-for-sound-recognition
├── cascaded-am-tuning-for-sound-recognition
│   ├── am-meta-analysis
│   │   └── cumulative
│   │       ├── Atencio2010
│   │       │   ├── txt files
│   │       │   └── ...
│   │       ├── Bartlett2007
│   │       └── (more directories...)
│   ├── ESC50
│   │   ├── ESC-50
│   │   │   ├── 101 - Dog
│   │       │   ├── ogg files
│   │       │   └── ...
│   │   │   ├── 102 - Rooster
│   │   │   └── (more directories...)
│   │   ├── info.txt
│   │   └── Results
│   │       ├── Result0
│   │       │   ├── Architecture.txt
│   │       │   ├── ConfusionMatrix.txt
│   │       │   ├── Params.txt
│   │       │   ├── Response
│   │       │   │   ├── Am
│   │       │   │   └── Am0
│   │       │   └── TrainedModel
│   │       ├── Result1
│   │       │   └── (same filenames as in Result0)
│   │       └── (more directories...)
│   └── TIMIT
│       ├── Results
│       │   ├── Result0
│       │   │   ├── Architecture.txt
│       │   │   ├── ConfusionMatrix.txt
│       │   │   ├── Params.txt
│       │   │   ├── Response
│       │   │   │   ├── Am
│       │   │   │   └── Am0
│       │   │   └── TrainedModel
│       │   ├── Result1
│       │   │   └── (same filenames as in Result0)
│       │   └── (more directories...)
│       └── TIMIT
│           ├── TEST
│           │   ├── DR1
│           │   ├── DR2
│           │   └── (more directories...)
│           └── TRAIN
│               ├── DR1
│               └── (more directories...)
├── draw_mtf_esc50.py
├── draw_mtf_timit.py
└── (more py files...)
```
  (It may be a bit confusing because directories with the same name appears twice...)
  
- ESC-50: Dataset for Environmental Sound Classification https://doi.org/10.7910/DVN/YDEPUT
  - Download audio files and put them in the directory "ESC50" indluced in the above dataset at figshare
  - Note: our code is built for the older version of ESC-50 with ogg format, in which folder organizations are slightly different from the current version at https://github.com/karoldvl/ESC-50
  
- TIMIT Acoustic-Phonetic Continuous Speech Corpus https://catalog.ldc.upenn.edu/LDC93S1
  - Download audio files and put them in the directory "TIMIT" indluced in the above dataset at figshare
## License
Please see [LICENSE](https://github.com/cycentum/cascaded-am-tuning-for-sound-recognition/blob/master/LICENSE).
