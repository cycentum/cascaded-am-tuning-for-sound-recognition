###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Takuya Koumura, Hiroki Terashima, Shigeto Furukawa. "Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition". bioRxiv. Cold Spring Harbor Laboratory; (2018): 308999.
###

import itertools
import chainer
from chainer import Variable, serializers, functions, optimizers
import numpy as np
from numpy import newaxis, float32, float64, int32, int64, int16, int8, uint8, uint32
from collections import defaultdict, Counter

try:
	import cupy
except:
	cupy=None
	pass

from model.net import totalInputLength, Net
from training.esc50.load_data import getLabels
# import Eve


def groupLabelWave(groupFolds, infos):
	groupLw=[defaultdict(list) for g in groupFolds]
	for ii,info in enumerate(infos):
		for glw,folds in zip(groupLw, groupFolds):
			if info[2] in folds: glw[info[0]].append(ii)
	return groupLw


def makeInpTru(labelWaveIndex, waves, remainingLabelWaveIndex, inputLength, labelSize, numLabel):
	x=np.empty((labelSize*numLabel, inputLength), float32)
	for li in range(numLabel):
		while len(remainingLabelWaveIndex[li])<labelSize: remainingLabelWaveIndex[li]=np.concatenate((remainingLabelWaveIndex[li],np.random.permutation(labelWaveIndex[li])),axis=0)
		for xi,it in enumerate(remainingLabelWaveIndex[li][:labelSize]):
			wi=it[0]
			ti=it[1]
			t0=ti-inputLength//2
			t1=t0+inputLength
			wave=waves[wi]
			if t0<0:
				if t1>len(wave): wave=np.concatenate((np.zeros(-t0, float32), wave, np.zeros(t1-len(wave), float32)))
				else: wave=np.concatenate((np.zeros(-t0, float32), wave[:t1]))
			else:
				if t1>len(wave): wave=np.concatenate((wave[t0:], np.zeros(t1-len(wave), float32)))
				else: wave=wave[t0:t1]
			x[li*labelSize+xi]=wave
		remainingLabelWaveIndex[li]=remainingLabelWaveIndex[li][labelSize:]
	tr=np.repeat(np.arange(numLabel, dtype=int32), labelSize)
	return x,tr


def findNumEpoch(architecture, waves, infos, gpu_id, waveFs):
	if cupy is not None and gpu_id>=0:
		xp=cupy
		cupy.cuda.Device(gpu_id).use()
	else: xp=np
	
	inputLength=totalInputLength(architecture)
	labels=getLabels()
	numLabel=len(labels)
	groupFold=((0,1,2),(3,),(4,))
	
	np.random.seed()
	seed=np.random.randint(0, np.iinfo(int32).max)
	np.random.seed(seed)
	net=Net(numLabel, architecture, functions.elu)
# 	opt=Eve(1e-4)
	opt=optimizers.Adam(1e-4)
	opt.setup(net)
	if gpu_id>=0: net.to_gpu(gpu_id)
	
	insLabelSize=2**2
	devSize=2**1
	devSegmentSecUpper=10
	
	devEpoch=2**5
	convergenceEpoch=2**5*devEpoch
	devSegmentLenUpper=int(devSegmentSecUpper*waveFs)
	devFold=sorted(set(groupFold[1]))
	devLabelWave=groupLabelWave((devFold,), infos)[0]
	devLabelWave=list(itertools.chain.from_iterable([[(li,i) for i in devLabelWave[la]] for li,la in enumerate(labels)]))
	devLabelWave=sorted(devLabelWave, key=lambda lw: len(waves[lw[1]]))
	devBatchIndex=np.array_split(np.arange(len(devLabelWave)), int(np.ceil(len(devLabelWave)/devSize)))
	devLabelSize=np.zeros(numLabel, int32)
	for li,wi in devLabelWave: devLabelSize[li]+=len(waves[wi])
	
	devWaves={}
	for li,wi in devLabelWave:
		wave=waves[wi]
		wave=np.concatenate((wave, np.zeros((inputLength-1)//2, float32)))
		devWaves[wi]=wave
	
	insFold=sorted(set(groupFold[0]))
	insLabelWave=groupLabelWave((insFold,), infos)[0]
	insLabelWaveIndex=[[] for i in range(len(labels))]
	for li,la in enumerate(labels):
		for i in insLabelWave[la]:
			wave=waves[i]
			timeIndex=np.arange(len(wave))
			waveIndex=np.ones(len(wave), int32)*i
			index=np.stack((waveIndex,timeIndex), axis=1)
			insLabelWaveIndex[li].append(index)
		insLabelWaveIndex[li]=np.concatenate(insLabelWaveIndex[li], axis=0)
	
	insRemainingLabelWave=[np.random.permutation(insLabelWaveIndex[li]) for li in range(len(labels))]
	
	epoch=0
	bestEpoch=0
	epochIncorrect={}
	while epoch<bestEpoch+convergenceEpoch:
		x,tr=makeInpTru(insLabelWaveIndex, waves, insRemainingLabelWave, inputLength, insLabelSize, numLabel)
		x=x[:,newaxis,:,newaxis]
		x=xp.asarray(x)
		x=Variable(x)
		x=net.callSingle(x, True)
		tr=tr[...,newaxis,newaxis]
		tr=xp.asarray(tr)
		e=functions.softmax_cross_entropy(x, tr)

		net.cleargrads()
		e.backward()
		e.unchain_backward()
# 		opt.update(loss=e.data)
		opt.update()
		
		if epoch%devEpoch!=devEpoch-1:
			epoch+=1
			continue
		incorrect=xp.zeros(numLabel, int32)
		with chainer.using_config("enable_backprop", False):
			for bi,index in enumerate(devBatchIndex):
				waveIndex=np.array([devLabelWave[i][1] for i in index])
				tru=np.array([devLabelWave[i][0] for i in index])
				waveLen=len(devWaves[waveIndex[-1]])
				segmentTimes=np.array_split(np.arange(waveLen), int(np.ceil((waveLen)/devSegmentLenUpper)))
				net.reset()
				for si,segTime in enumerate(segmentTimes):
					t0=segTime[0]
					t1=segTime[-1]+1
					x=np.zeros((len(index), t1-t0), float32)
					tr=-np.ones((len(index), t1-t0), int32)
					for xi,wi in enumerate(waveIndex):
						if len(devWaves[wi])<=t0: continue
						w=devWaves[wi][t0:t1]
						x[xi, :len(w)]=w
						tr[xi, :len(w)]=tru[xi]
					if t0<(inputLength-1)//2: tr[:,:(inputLength-1)//2-t0]=-1
					
					x=x[:,newaxis,:,newaxis]
					x=xp.asarray(x)
					x=Variable(x)
					x=net(x, False)
					x.unchain_backward()
					
					x=xp.argmax(x.data, axis=1)
					tr=tr[...,newaxis]
					tr=xp.asarray(tr)
					for li,la in enumerate(labels): incorrect[li]+=(x[tr==li]!=li).sum()
	
			net.reset()
			if gpu_id>=0: incorrect=cupy.asnumpy(incorrect)
			incorrect=(incorrect/devLabelSize).mean()
			print("epoch", epoch, "incorrect", incorrect)
		
		if len(epochIncorrect)==0 or incorrect<epochIncorrect[bestEpoch]: bestEpoch=epoch
		epochIncorrect[epoch]=incorrect
		epoch+=1
	
	devEpochs=np.array(sorted(epochIncorrect), int32)
	bestScore=epochIncorrect[bestEpoch]
	epochIncorrect=np.array([epochIncorrect[ep] for ep in devEpochs])
	
	return bestEpoch, bestScore, seed
	

def train(architecture, waves, infos, gpu_id, waveFs, numEpoch, seed):
	if cupy is not None and gpu_id>=0:
		xp=cupy
		cupy.cuda.Device(gpu_id).use()
	else: xp=np
	
	inputLength=totalInputLength(architecture)
	labels=getLabels()
	numLabel=len(labels)
	groupFold=((0,1,2),(3,),(4,))
	
	insLabelSize=2**2
	
	np.random.seed(seed)
	net=Net(numLabel, architecture, functions.elu)
# 	opt=Eve(1e-4)
	opt=optimizers.Adam(1e-4)
	opt.setup(net)
	if gpu_id>=0: net.to_gpu(gpu_id)

	insFold=set(itertools.chain.from_iterable(groupFold[:2]))
	insLabelWave=groupLabelWave((insFold,), infos)[0]
	insLabelWaveIndex=[[] for i in range(len(labels))]
	for li,la in enumerate(labels):
		for i in insLabelWave[la]:
			wave=waves[i]
			timeIndex=np.arange(len(wave))
			waveIndex=np.ones(len(wave), int32)*i
			index=np.stack((waveIndex,timeIndex), axis=1)
			insLabelWaveIndex[li].append(index)
		insLabelWaveIndex[li]=np.concatenate(insLabelWaveIndex[li], axis=0)
	
	insRemainingLabelWave=[np.random.permutation(insLabelWaveIndex[li]) for li in range(len(labels))]

	for epoch in range(numEpoch):
		print("Training: Epoch", epoch, "/", numEpoch)
		
		x,tr=makeInpTru(insLabelWaveIndex, waves, insRemainingLabelWave, inputLength, insLabelSize, numLabel)
		x=x[:,newaxis,:,newaxis]
		x=xp.asarray(x)
		x=Variable(x)
		x=net.callSingle(x, True)
		tr=tr[...,newaxis,newaxis]
		tr=xp.asarray(tr)
		e=functions.softmax_cross_entropy(x, tr)

		net.cleargrads()
		e.backward()
		e.unchain_backward()
		opt.update(loss=e.data)
# 		opt.update()
	
	return net


def evaluate(architecture, waves, infos, gpu_id, waveFs, fileParam):
	if cupy is not None and gpu_id>=0:
		xp=cupy
		cupy.cuda.Device(gpu_id).use()
	else: xp=np
	
	inputLength=totalInputLength(architecture)
	labels=getLabels()
	numLabel=len(labels)
	groupFold=((0,1,2),(3,),(4,))

	devSize=2**1
	devSegmentSecUpper=10

	net=Net(numLabel, architecture, functions.elu)
	serializers.load_hdf5(fileParam, net)
	if gpu_id>=0: net.to_gpu(gpu_id)
	
	devSegmentLenUpper=int(devSegmentSecUpper*waveFs)
	devFold=sorted(set(groupFold[2]))
	devLabelWave=groupLabelWave((devFold,), infos)[0]
	devLabelWave=list(itertools.chain.from_iterable([[(li,i) for i in devLabelWave[la]] for li,la in enumerate(labels)]))
	devLabelWave=sorted(devLabelWave, key=lambda lw: len(waves[lw[1]]))
	devBatchIndex=np.array_split(np.arange(len(devLabelWave)), int(np.ceil(len(devLabelWave)/devSize)))
	devLabelSize=np.zeros(numLabel, int32)
	for li,wi in devLabelWave: devLabelSize[li]+=len(waves[wi])
	
	devWaves={}
	for li,wi in devLabelWave:
		wave=waves[wi]
		wave=np.concatenate((wave, np.zeros((inputLength-1)//2, float32)))
		devWaves[wi]=wave
	
	with chainer.using_config("enable_backprop", False):
		confusion=np.zeros((numLabel,numLabel), int32)
		for bi,index in enumerate(devBatchIndex):
			waveIndex=np.array([devLabelWave[i][1] for i in index])
			tru=np.array([devLabelWave[i][0] for i in index])
			waveLen=len(devWaves[waveIndex[-1]])
			segmentTimes=np.array_split(np.arange(waveLen), int(np.ceil((waveLen)/devSegmentLenUpper)))
			net.reset()
			for si,segTime in enumerate(segmentTimes):
				t0=segTime[0]
				t1=segTime[-1]+1
				x=np.zeros((len(index), t1-t0), float32)
				tr=-np.ones((len(index), t1-t0), int32)
				for xi,wi in enumerate(waveIndex):
					if len(devWaves[wi])<=t0: continue
					w=devWaves[wi][t0:t1]
					x[xi, :len(w)]=w
					tr[xi, :len(w)]=tru[xi]
				if t0<(inputLength-1)//2: tr[:,:(inputLength-1)//2-t0]=-1
				
				x=x[:,newaxis,:,newaxis]
				x=xp.asarray(x)
				x=Variable(x)
				x=net(x, False)
				x.unchain_backward()
				
				x=xp.argmax(x.data, axis=1)
				if gpu_id>=0: x=cupy.asnumpy(x)
				x=x.flatten()
				tr=tr.flatten()
				for xi,ti in zip(x[tr>=0],tr[tr>=0]): confusion[ti,xi]+=1
		
		net.reset()
		assert (np.sum(confusion, axis=1)==devLabelSize).all()
		return confusion

		
def compTrainingRms(waves, infos):
	groupFold=((0,1,2),(3,),(4,))
	insFold=set(itertools.chain.from_iterable(groupFold[:2]))
	rms=[]
	for ii,info in enumerate(infos):
		if info[2] in insFold:
			wave=waves[ii]
			rms.append((wave**2).mean()**0.5)
	return np.array(rms).mean()
