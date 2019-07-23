###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Koumura T, Terashima H, Furukawa S (2019) Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition. J Neurosci 39(28):5517–5533.
###

import numpy as np
from pathlib import Path
from chainer import serializers

from model.architecture import sampleArchitecture, architectureStr
from training.esc50.train_eval import findNumEpoch, train, evaluate
from training.esc50.load_data import loadWaves, readInfos, fade
from utils.utils import localTimeStr

if __name__=="__main__":
	dirRoot=Path("./cascaded-am-tuning-for-sound-recognition")
	gpu_id=0

	dirEsc=dirRoot/"ESC50"
	fileInfo=dirEsc/"info.txt"
	dirSound=dirEsc/"ESC-50"

	infos=readInfos(str(fileInfo))
	waves, waveFs=loadWaves(str(dirSound), infos)
	waves=fade(waves, waveFs)

	dirResult=dirEsc/"Results"/("Result"+localTimeStr())
	dirResult.mkdir(exist_ok=True, parents=True)

	#sample architecture
	numLayer=13
	totalInputLenUpper=8192
	numChannel=128
	filterLenUpper=8
	architecture=sampleArchitecture(numLayer, totalInputLenUpper, numChannel, filterLenUpper)
	print("Architecture", architecture)
	fileArchitecture=dirResult/"Architecture.txt"
	with open(fileArchitecture, "w") as f:
		print(architectureStr(architecture), file=f, sep="\t")

	#find num epoch
	numEpoch, bestScore, seed=findNumEpoch(architecture, waves, infos, gpu_id, waveFs)
	print("numEpoch", numEpoch, "bestScore", bestScore, "seed", seed)
	fileParams=dirResult/"Params.txt"
	with open(fileParams, "w") as f:
		print("NumEpoch", numEpoch, file=f, sep="\t")
		print("BestScore", bestScore, file=f, sep="\t")
		print("Seed", seed, file=f, sep="\t")

	#tarining
	net=train(architecture, waves, infos, gpu_id, waveFs, numEpoch, seed)
	fileParam=dirResult/"TrainedModel"
	serializers.save_hdf5(str(fileParam), net)
	print("Saved trained param in", str(fileParam))

	#evaluation
	confusion=evaluate(architecture, waves, infos, gpu_id, waveFs, fileParam)
	fileConfusion=dirResult/"ConfusionMatrix.txt"
	np.savetxt(fileConfusion, confusion, "%d", "\t")

	correct=confusion/np.sum(confusion, axis=1)
	correct=np.mean(np.diag(correct))
	print("Correct rate", correct)
