###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Koumura T, Terashima H, Furukawa S (2019) Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition. J Neurosci 39(28):5517–5533.
###

import pickle
import soundfile
import numpy as np
from numpy import newaxis, float32, float64, int32, int64, int16, int8, uint8, uint32

from utils.utils import readTable


def readInfos(file):
	'''
	@return: [(label, waveName, fold, (t0, t1)), ...]
	'''
	bounds=readTable(file)
	infos=[]
	for b in bounds:
		if b[-1]=="s": continue
		label=b[0]
		waveName=b[1]
		fold=int(waveName.split("-")[0])-1
		t0=int(b[3])
		t1=int(b[4])
		infos.append((label, waveName, fold, (t0,t1)))
	return infos


def toMap(iterable, keyFunc, valueFunc=lambda x: x, forceList=False, assertSingle=False):
	d={}
	if not forceList: count={}
	for e in iterable:
		k=keyFunc(e)
		v=valueFunc(e)
		if assertSingle: assert k not in d
		if forceList:
			if k not in d: d[k]=[]
			d[k].append(v)
		else:
			if k not in d:
				d[k]=v
				count[k]=1
			else:
				if count[k]==1:
					d[k]=[d[k],v]
					count[k]=2
				else: d[k].append(v)
	return d


def loadWaves(dirSound, infos):
	waveInfos=toMap(infos, lambda x: x[1], forceList=True)
	infoWave={}
	waveFs=None
	for waveName,inf in waveInfos.items():
		print("Loading sound", waveName)
		label=inf[0][0]
		wave,fs=soundfile.read(dirSound+"/"+label+"/"+waveName+".ogg", dtype=float32)
		for i in inf: infoWave[i]=wave[i[3][0]:i[3][1]]
		if waveFs is not None: assert waveFs==fs
		waveFs=fs
	assert len(infoWave)==len(infos)
	waves=[infoWave[i] for i in infos]
	return waves, waveFs


def fade(waves, waveFs):
	fadeSec=0.01
	fadeLen=int(fadeSec*waveFs)
	win=np.hanning(fadeLen*2)
	for wi,wave in enumerate(waves):
		if len(wave)>fadeLen:
			wave[:fadeLen//2]*=win[:fadeLen//2]
			wave[-fadeLen//2:]*=win[-fadeLen//2:]
		else:
			wave[:len(wave)//2]*=win[:len(wave)//2]
			wave[len(wave)//2:]*=win[-len(wave[len(wave)//2:]):]
	return waves


def getLabels():
	nonHumanLabels=( #size=18
	"101 - Dog",
	"102 - Rooster",
	"103 - Pig",
	"104 - Cow",
	"105 - Frog",
	"106 - Cat",
	"107 - Hen",
	"108 - Insects",
	"109 - Sheep",
	"110 - Crow",
	"201 - Rain",
	"202 - Sea waves",
	"203 - Crackling fire",
	"204 - Crickets",
	"205 - Chirping birds",
	"206 - Water drops",
	"207 - Wind",
	"210 - Thunderstorm",)
	return nonHumanLabels
