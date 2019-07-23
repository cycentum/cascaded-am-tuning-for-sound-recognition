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
from training.timit.load_data import loadData
from physiology.recording import compNoiseAm0AveSyn, compNoiseAmAveSyn, compSilenceAveSyn, compToneAveSyn
from training.timit.train_eval import compTrainingRms


if __name__=="__main__":
	dirRoot=Path("./cascaded-am-tuning-for-sound-recognition")
	gpu_id=1

	dirTimit=dirRoot/"TIMIT"
	fileInfo=dirTimit/"info.txt"
	dirSound=dirTimit/"TIMIT"
	infos, waves, trues, labels, waveFs=loadData(dirSound)
	trainingRms=compTrainingRms(waves, infos)

	dirResult=dirTimit/"Results"/"Result22"

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
