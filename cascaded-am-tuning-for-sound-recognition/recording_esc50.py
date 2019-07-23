###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Koumura T, Terashima H, Furukawa S (2019) Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition. J Neurosci 39(28):5517–5533.
###

import numpy as np
import pickle
from pathlib import Path

from model.architecture import readArchitecture
from training.esc50.load_data import loadWaves, readInfos, fade
from physiology.recording import compNoiseAm0AveSyn, compNoiseAmAveSyn, compSilenceAveSyn, compToneAveSyn
from training.esc50.train_eval import compTrainingRms


if __name__=="__main__":
	dirRoot=Path("./cascaded-am-tuning-for-sound-recognition")
	gpu_id=0

	dirEsc=dirRoot/"ESC50"
	fileInfo=dirEsc/"info.txt"
	dirSound=dirEsc/"ESC-50"
	infos=readInfos(str(fileInfo))
	waves, waveFs=loadWaves(str(dirSound), infos)
	waves=fade(waves, waveFs)
	trainingRms=compTrainingRms(waves, infos)

	dirResult=dirEsc/"Results"/"Result11"

	fileArchitecture=dirResult/"Architecture.txt"
	architecture=readArchitecture(fileArchitecture)

	fileModel=dirResult/"TrainedModel"
	stimSec=8

	dirResponse=dirResult/"Response"
	dirResponse.mkdir(exist_ok=True, parents=True)

	response=compNoiseAmAveSyn(stimSec, waveFs, fileModel, architecture, gpu_id, trainingRms)
	fileResponse=dirResponse/"Am"
	with open(fileResponse, "wb") as f: pickle.dump(response, f)

	response=compNoiseAm0AveSyn(stimSec, waveFs, fileModel, architecture, gpu_id, trainingRms)
	fileResponse=dirResponse/"Am0"
	with open(fileResponse, "wb") as f: pickle.dump(response, f)
