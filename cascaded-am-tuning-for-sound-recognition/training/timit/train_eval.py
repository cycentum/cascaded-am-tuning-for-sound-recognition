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
# import Eve


def getCoreTestSpeaker():
	CORE_TEST_SPEAKER=(
	"MDAB0",
	"MWBT0",
	"FELC0",
	"MTAS1",
	"MWEW0",
	"FPAS0",
	"MJMP0",
	"MLNT0",
	"FPKT0",
	"MLLL0",
	"MTLS0",
	"FJLM0",
	"MBPM0",
	"MKLT0",
	"FNLP0",
	"MCMJ0",
	"MJDH0",
	"FMGD0",
	"MGRT0",
	"MNJM0",
	"FDHC0",
	"MJLN0",
	"MPAM0",
	"FMLD0",
	)
	return CORE_TEST_SPEAKER


def coreTestIndex(infos):
	CORE_TEST_SPEAKER=getCoreTestSpeaker()
	index=[]
	for ii,info in enumerate(infos):
		if info[0]=="TEST" and (info[2] in CORE_TEST_SPEAKER) and info[3][:2]!="SA": index.append(ii)
	return index


def traGroupIndex(infos, numTraGroup):
	speakers=sorted(list(set([info[2] for info in filter(lambda i: i[0]=="TRAIN", infos)])))
	speakerGroups=np.array_split(np.random.permutation(len(speakers)), numTraGroup)
	index=[]
	for gr in speakerGroups:
		index.append([])
		spSet=set([speakers[si] for si in gr])
		for ii,info in enumerate(infos):
			if info[2] in spSet and info[3][:2]!="SA": index[-1].append(ii)
	return index


def makeLabelIndexTime(index, labels, trues):
	insIndex=np.array(index)
	insLabelIndexTime=[[] for li,la in enumerate(labels)]
	for i in insIndex:
		tr=trues[i]
		times=np.arange(len(tr))
		for li,la in enumerate(labels):
			t=times[tr==li]
			it=np.stack((i*np.ones(len(t), int32), t)).T
			insLabelIndexTime[li].append(it)
	for li,la in enumerate(labels): insLabelIndexTime[li]=np.concatenate(insLabelIndexTime[li],axis=0)
	return insLabelIndexTime


def makeInpTru(labels, insLabelSize, inputLength, remainingInsLabelIndexTime, waves, trues):
	center=False
	x=np.zeros((len(labels)*insLabelSize, inputLength), float32)
	tr=np.zeros((len(labels)*insLabelSize), int32)
	for li,lit in enumerate(remainingInsLabelIndexTime):
		for xi,it in enumerate(lit[:insLabelSize]):
			wi=it[0]
			ti=it[1]
			if center:
				t0=ti-inputLength//2
				t1=t0+inputLength
				w=waves[wi][max(t0,0):min(t1,len(waves[wi]))]
				if t0<0: w=np.concatenate((np.zeros(-t0, float32),w))
				if t1>len(waves[wi]): w=np.concatenate((w, np.zeros(t1-len(waves[wi]), float32)))
			else:
				w=waves[wi][ti-inputLength+1:ti+1]
				if len(w)<inputLength: w=np.concatenate((np.zeros(inputLength-len(w), float32), w))
			x[li*insLabelSize+xi]=w
			tr[li*insLabelSize+xi]=trues[wi][ti]
		remainingInsLabelIndexTime[li]=remainingInsLabelIndexTime[li][insLabelSize:]
	return x, tr


def findNumEpoch(architecture, waves, trues, labels, infos, gpu_id, waveFs):
	if cupy is not None and gpu_id>=0:
		xp=cupy
		cupy.cuda.Device(gpu_id).use()
	else: xp=np
	
	valIndex=coreTestIndex(infos)
	np.random.seed(0)
	insIndex, devIndex=traGroupIndex(infos, 2)
	insIndex=np.array(insIndex)
	insLabelIndexTime=makeLabelIndexTime(insIndex, labels, trues)
	
	insLabelSize=2**2
	devEpoch=2**5
	convergenceEpoch=2**5*devEpoch
	
	devBatchSizeUpper=2**8
	devSegmentSecUpper=0.1
	devSegmentLenUpper=int(devSegmentSecUpper*waveFs)
	
	devIndex=sorted(devIndex, key=lambda i: len(waves[i]))
	devIndex=np.array(devIndex)
	devBatchIndex=np.array_split(devIndex, int(np.ceil(len(devIndex)/devBatchSizeUpper)))
	devLabelSize=np.zeros(len(labels), int32)
	for i in devIndex:
		for li,la in enumerate(labels): devLabelSize[li]+=(trues[i]==li).sum()
	
	inputLength=totalInputLength(architecture)
	
	np.random.seed()
	seed=np.random.randint(0, np.iinfo(int32).max)
	np.random.seed(seed)
	
	net=Net(len(labels), architecture, functions.elu)
	opt=optimizers.Adam(1e-4)
# 	opt=Eve(1e-4)
	opt.setup(net)
	if gpu_id>=0: net.to_gpu(gpu_id)
	
	remainingInsLabelIndexTime=[np.random.permutation(lt) for lt in insLabelIndexTime]
	
	epoch=0
	bestEpoch=0
	epochIncorrect={}
	while epoch<bestEpoch+convergenceEpoch:
		for li,lit in enumerate(remainingInsLabelIndexTime):
			if len(lit)<insLabelSize: remainingInsLabelIndexTime[li]=np.concatenate((lit, np.random.permutation(insLabelIndexTime[li])))
		x,tr=makeInpTru(labels, insLabelSize, inputLength, remainingInsLabelIndexTime, waves, trues)
		
		x=x[:,newaxis,:,newaxis]
		x=xp.asarray(x)
		x=Variable(x)
		x=net.callSingle(x, True)
		tr=tr[...,newaxis,newaxis]
		tr=xp.asarray(tr)
		e=functions.softmax_cross_entropy(x, tr, normalize=True)

		net.cleargrads()
		e.backward()
		e.unchain_backward()
		opt.update()
# 		opt.update(loss=e.data)
		
		if epoch%devEpoch!=devEpoch-1:
			epoch+=1
			continue
		incorrect=xp.zeros(len(labels), int32)
		with chainer.using_config("enable_backprop", False):
			for index in devBatchIndex:
				waveLen=len(waves[index[-1]])
				segmentTimes=np.array_split(np.arange(waveLen), int(np.ceil(waveLen/devSegmentLenUpper)))
				net.reset()
				for si,segTime in enumerate(segmentTimes):
					t0=segTime[0]
					t1=segTime[-1]+1
					x=np.zeros((len(index), t1-t0), float32)
					tr=-np.ones((len(index), t1-t0), int32)
					for xi,wi in enumerate(index):
						if len(waves[wi])>t0:
							w=waves[wi][t0:t1]
							x[xi, :len(w)]=w
						if len(waves[wi])>t0: tr[xi, :len(w)]=trues[wi][t0:t1]
						
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
			if cupy is not None: incorrect=cupy.asnumpy(incorrect)
			incorrect=(incorrect/devLabelSize).mean()
			print("epoch", epoch, "incorrect", incorrect)
			
			if len(epochIncorrect)==0 or incorrect<min([epochIncorrect[ep] for ep in epochIncorrect]): bestEpoch=epoch
			epochIncorrect[epoch]=incorrect
			epoch+=1
	
	devEpochs=np.array(sorted(epochIncorrect), int32)
	epochIncorrect=np.array([epochIncorrect[ep] for ep in devEpochs])
	bestIncorrect=epochIncorrect.min()
	
	return bestEpoch, bestIncorrect, seed
	

def train(architecture, waves, trues, labels, infos, gpu_id, waveFs, numEpoch, seed):
	if cupy is not None and gpu_id>=0:
		xp=cupy
		cupy.cuda.Device(gpu_id).use()
	else: xp=np
	
	valIndex=coreTestIndex(infos)
	np.random.seed(0)
	insIndex,=traGroupIndex(infos, 1)
	insIndex=np.array(insIndex)
	insLabelIndexTime=makeLabelIndexTime(insIndex, labels, trues)
	
	insLabelSize=2**2 #la12 tot4096 ch128
		
	inputLength=totalInputLength(architecture)
	
	np.random.seed(seed)
	net=Net(len(labels), architecture, functions.elu)
	opt=optimizers.Adam(1e-4)
# 	opt=Eve(1e-4)
	opt.setup(net)
	if gpu_id>=0: net.to_gpu(gpu_id)
	
	remainingInsLabelIndexTime=[np.random.permutation(lt) for lt in insLabelIndexTime]
	for epoch in range(numEpoch):
		print("Training: Epoch", epoch, "/", numEpoch)
		for li,lit in enumerate(remainingInsLabelIndexTime):
			if len(lit)<insLabelSize: remainingInsLabelIndexTime[li]=np.concatenate((lit, np.random.permutation(insLabelIndexTime[li])))
		x,tr=makeInpTru(labels, insLabelSize, inputLength, remainingInsLabelIndexTime, waves, trues)
		
		x=x[:,newaxis,:,newaxis]
		x=xp.asarray(x)
		x=Variable(x)
		x=net.callSingle(x, True)
		tr=tr[...,newaxis,newaxis]
		tr=xp.asarray(tr)
		e=functions.softmax_cross_entropy(x, tr, normalize=True)

		net.cleargrads()
		e.backward()
		e.unchain_backward()
		opt.update()
# 		opt.update(loss=e.data)
			
	return net


def evaluate(architecture, waves, trues, labels, infos, gpu_id, waveFs, fileParam):
	if cupy is not None and gpu_id>=0:
		xp=cupy
		cupy.cuda.Device(gpu_id).use()
	else: xp=np
	
	valIndex=coreTestIndex(infos)
	
	devBatchSizeUpper=2**8
	devSegmentSecUpper=0.1
	devSegmentLenUpper=int(devSegmentSecUpper*waveFs)
	
	devIndex=sorted(valIndex, key=lambda i: len(waves[i]))
	devIndex=np.array(devIndex)
	devBatchIndex=np.array_split(devIndex, int(np.ceil(len(devIndex)/devBatchSizeUpper)))
	devLabelSize=np.zeros(len(labels), int32)
	for i in devIndex:
		for li,la in enumerate(labels): devLabelSize[li]+=(trues[i]==li).sum()
		
	net=Net(len(labels), architecture, functions.elu)
	serializers.load_hdf5(fileParam, net)
	if gpu_id>=0: net.to_gpu(gpu_id)
	inputLength=totalInputLength(architecture)
		
	with chainer.using_config("enable_backprop", False):
		confusion=np.zeros((len(labels),len(labels)), int32)
		for index in devBatchIndex:
			waveLen=len(waves[index[-1]])
			segmentTimes=np.array_split(np.arange(waveLen), int(np.ceil(waveLen/devSegmentLenUpper)))
			net.reset()
			for si,segTime in enumerate(segmentTimes):
				t0=segTime[0]
				t1=segTime[-1]+1
				x=np.zeros((len(index), t1-t0), float32)
				tr=-np.ones((len(index), t1-t0), int32)
				for xi,wi in enumerate(index):
					if len(waves[wi])>t0:
						w=waves[wi][t0:t1]
						x[xi, :len(w)]=w
					if len(waves[wi])>t0: tr[xi, :len(w)]=trues[wi][t0:t1]
					
				x=x[:,newaxis,:,newaxis]
				x=xp.asarray(x)
				x=Variable(x)
				x=net(x, False)
				
				x=xp.argmax(x.data, axis=1)
				if cupy is not None: x=cupy.asnumpy(x)
				x=x.flatten()
				tr=tr.flatten()
				for xi,ti in zip(x,tr):
					if ti>=0: confusion[ti,xi]+=1
		
		assert (np.sum(confusion, axis=1)==devLabelSize).all()
		return confusion
		
		
def compTrainingRms(waves, infos):
	insIndex,=traGroupIndex(infos, 1)
	insIndex=np.array(insIndex)
	rms=np.empty(len(insIndex))
	for iii,ii in enumerate(insIndex):
		wave=waves[ii]
		rms[iii]=(wave**2).mean()**0.5
	return rms.mean() #We used (this value * 0.57) in the paper