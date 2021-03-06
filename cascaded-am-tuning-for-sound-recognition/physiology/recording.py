###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Koumura T, Terashima H, Furukawa S (2019) Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition. J Neurosci 39(28):5517–5533.
###

import numpy as np
from numpy import newaxis, float32, float64, int32, int64, int16, int8, uint8, uint32
import chainer
from chainer import Variable, serializers, functions, links, optimizers, cuda, Chain

from model.net import Net, totalInputLength

try:
	import cupy
except:
	cupy=None
	pass


def loadNet(architecture, file):
	net=Net(0, architecture, functions.elu)
	serializers.load_hdf5(file, net)
	return net


def vectorCosSin(freq, inputLen, waveLen, waveFs):
	times=np.arange(waveLen)/waveFs
	period=1/freq
	angle=times%period/period*2*np.pi
	cos=np.cos(angle)
	sin=np.sin(angle)

	cos=cos[inputLen-1:]
	sin=sin[inputLen-1:]
	return cos,sin


def scaleRms(waves, targetRms):
	waves*=targetRms/(waves**2).mean(axis=-1, keepdims=True)**0.5
	return waves


def compRepresentation(net, x):
	'''
	Deprecated.
	Only used in compLongRepresentation(), which is also deprecated.
	
	@param x: shape=(batch, channel=1, length, 1)
	@return repre: len=layer, [shape=(batch, channel, length, 1),... ]
	'''
	train=False
	if cupy is not None: xp=cupy.get_array_module(x.data)
	else: xp=np
	numBatch=x.shape[0]
	repre=[]
	for li,st in enumerate(net.structure):
		numChannel,inputLen,filterLen=st
		prevLen=inputLen-1
		p=net.prev[li]
		if x.shape[2]>prevLen: net.prev[li]=xp.split(x.data, (x.shape[2]-prevLen,), axis=2)[1]
		elif p is not None:
			net.prev[li]=xp.concatenate((p, x.data), axis=2)
			if net.prev[li].shape[2]>prevLen: net.prev[li]=xp.split(net.prev[li], (net.prev[li].shape[2]-prevLen,), axis=2)[1]
		else: net.prev[li]=x.data
		if p is None: p=xp.zeros((numBatch, 1 if li==0 else net.structure[li-1][0], prevLen, 1), float32)
		if p.shape[2]<prevLen: p=xp.concatenate((xp.zeros((p.shape[0], p.shape[1], prevLen-p.shape[2], p.shape[3]), float32), p), axis=2)
		p=Variable(p)
		x=functions.concat((p, x), axis=2)

		x=net["c"+str(li)](x)
# 		x=functions.elu(x)
		x=net.act(x)
		repre.append(x.data)

	return repre


def compRepresentationSingle(net, x):
	'''
	@param x: shape=(batch, channel=1, length, 1)
	@return repre: len=layer, [shape=(batch, channel, length, 1),... ]
	'''
	repre=[]
	
	for li,st in enumerate(net.structure):
		x=net["c"+str(li)](x)
		x=net.act(x)
		repre.append(x.data)
	
	return repre


def compLongRepresentation(net, waves, segmentLenUpper, waveLen, xp, trimInputLen=True):
	'''
	Deprecated.
	compLongRepresentationSingle() returns the same result when trimInputLen=True
	
	'''
	segmentTimes=np.array_split(np.arange(waveLen), int(np.ceil(waveLen/segmentLenUpper)))
	segmentTimes=[(x[0],x[-1]+1) for x in segmentTimes]
	net.reset()
	repre=None
	for si,(t0,t1) in enumerate(segmentTimes):
		x=waves[:, t0:t1]
		x=x[:,newaxis,:,newaxis]
		x=xp.asarray(x, float32)
		x=Variable(x)
		r=compRepresentation(net, x)
		r=xp.stack(r, axis=0)
		r=r[...,0]
		if repre is None: repre=np.empty((r.shape[0], len(waves), r.shape[2], waveLen), r.dtype)
		if xp!=np: r=cupy.asnumpy(r)
		repre[...,t0:t1]=r

	if trimInputLen:
		inputLen=totalInputLength(net.structure)
		repre=repre[...,inputLen-1:]

	return repre


def compLongRepresentationSingle(net, waves, xp, trimInputLen=True):
	net.reset()
	
	x=waves
	batchSize, waveLen=x.shape
	
	inputLen=totalInputLength(net.structure)
	x=np.concatenate((np.zeros((batchSize, inputLen-1), x.dtype), x), axis=1)
	
	x=x[:,newaxis,:,newaxis]
	x=xp.asarray(x, float32)
	x=Variable(x)
	layerRepre=compRepresentationSingle(net, x)
	for li,r in enumerate(layerRepre):
		r=r[...,0]
		if xp!=np: r=cupy.asnumpy(r)
		
		r=r[..., -waveLen:]
		if trimInputLen:
			r=r[...,inputLen-1:]
		layerRepre[li]=r

	return layerRepre


def compToneAveSyn(stimSec, waveFs, fileModel, architecture, gpu_id, trainingRms):
	with chainer.using_config("enable_backprop", False):
		if cupy is not None and gpu_id>=0:
			xp=cupy
			cupy.cuda.Device(gpu_id).use()
		else: xp=np

		waveLen=int(stimSec*waveFs)
		batchSizeUpper=1

		times=np.arange(waveLen)/waveFs
		freqs=np.logspace(np.log10(100), np.log10(5000), 2**8)
		ampScale=np.logspace(np.log10(1/512), np.log10(1), 2**8)

		batchAmps=np.array_split(ampScale, int(np.ceil(len(ampScale)/batchSizeUpper)))
		segmentSecUpper=2
		segmentLenUpper=int(segmentSecUpper*waveFs)

		net=loadNet(architecture, fileModel)
		if gpu_id>=0: net.to_gpu(gpu_id)
		inputLen=totalInputLength(net.structure)

		freqResponse=[]
		for fi,freq in enumerate(freqs):
			cos,sin=vectorCosSin(freq, inputLen, waveLen, waveFs)
			batchResponse=[]
			for bi,ba in enumerate(batchAmps):
				waves=np.sin(freq*2*np.pi*times)
				waves=scaleRms(waves, trainingRms)
				waves=waves*ba[:,newaxis] #shape=(amp, length)
# 				repre=compLongRepresentation(net, waves, segmentLenUpper, waveLen, xp) #compLongRepresentationSingle() returns the same result when trimInputLen=True
				repre=compLongRepresentationSingle(net, waves, xp)

				repre+=1 #elu
				ave=repre.mean(axis=-1)
				s=repre.sum(axis=-1)
				syn=(((repre*cos).sum(axis=-1)/s)**2+((repre*sin).sum(axis=-1)/s)**2)**0.5
				syn[s==0]=0
				resp=np.stack((ave,syn), axis=0) #shape=(type, layer, amp, channel)

				batchResponse.append(resp)
			batchResponse=np.concatenate(batchResponse, axis=-2) #shape=(type, layer, amp, channel)
			freqResponse.append(batchResponse)
		freqResponse=np.array(freqResponse) #shape=(freq, type, layer, amp, channel)

		return freqResponse


def compSilenceAveSyn(stimSec, waveFs, fileModel, architecture, gpu_id):
	with chainer.using_config("enable_backprop", False):
		if cupy is not None and gpu_id>=0:
			xp=cupy
			cupy.cuda.Device(gpu_id).use()
		else: xp=np

		waveLen=int(stimSec*waveFs)

		freqs=np.logspace(np.log10(100), np.log10(5000), 2**8)

		segmentSecUpper=2
		segmentLenUpper=int(segmentSecUpper*waveFs)

		net=loadNet(architecture, fileModel)
		if gpu_id>=0: net.to_gpu(gpu_id)
		inputLen=totalInputLength(net.structure)

		waves=np.zeros((1,waveLen), float32)
# 		repre=compLongRepresentation(net, waves, segmentLenUpper, waveLen, xp) #shape=(layer, 1, channel, length) #compLongRepresentationSingle() returns the same result when trimInputLen=True
		repre=compLongRepresentationSingle(net, waves, xp)
		repre+=1 #elu
		repre=repre[:,0,:,:] #shape=(layer, channel, length)

		ave=repre.mean(axis=-1) #shape=(layer, channel)

		freqResponse=[]
		for fi,freq in enumerate(freqs):
			cos,sin=vectorCosSin(freq, inputLen, waveLen, waveFs)

			s=repre.sum(axis=-1)
			syn=(((repre*cos).sum(axis=-1)/s)**2+((repre*sin).sum(axis=-1)/s)**2)**0.5
			syn[s==0]=0
			freqResponse.append(syn)
		freqResponse=np.array(freqResponse) #shape=(freq, layer, channel)

		return ave, freqResponse


def compNoiseAmAveSyn(stimSec, waveFs, fileModel, architecture, gpu_id, trainingRms):
	with chainer.using_config("enable_backprop", False):
		if cupy is not None and gpu_id>=0:
			xp=cupy
			cupy.cuda.Device(gpu_id).use()
		else: xp=np

		modDepth=1
		waveLen=int(stimSec*waveFs)

		np.random.seed(0)

		times=np.arange(waveLen)/waveFs
		freqs=np.logspace(np.log10(1), np.log10(2000), 2**8)
		meanSize=2**2 #we used 2**4 in our paper

		batchSizeUpper=meanSize
		batchSize=np.array_split(np.arange(meanSize), int(np.ceil(meanSize/batchSizeUpper)))
		batchSize=[len(x) for x in batchSize]
		segmentSecUpper=2 #decrease this for a GPU with smaller memory
		segmentLenUpper=int(segmentSecUpper*waveFs)

		net=loadNet(architecture, fileModel)
		if gpu_id>=0: net.to_gpu(gpu_id)
		inputLen=totalInputLength(net.structure)

		freqResponse=[]
		for fi, freq in enumerate(freqs):
			print("Conducting physiology:", "stimulus AM freq:", freq)
			cos,sin=vectorCosSin(freq, inputLen, waveLen, waveFs)

			batchResponse=[]
			for bi,bs in enumerate(batchSize):
				waves=np.random.randn(bs, waveLen)
				waves*=(1-modDepth*np.cos(freq*2*np.pi*times))
				waves=scaleRms(waves, trainingRms)
# 				repre=compLongRepresentation(net, waves, segmentLenUpper, waveLen, xp) #compLongRepresentationSingle() returns the same result when trimInputLen=True
				repre=compLongRepresentationSingle(net, waves, xp)

				repre+=1 #elu
				ave=repre.mean(axis=-1)
				s=repre.sum(axis=-1)
				syn=(((repre*cos).sum(axis=-1)/s)**2+((repre*sin).sum(axis=-1)/s)**2)**0.5
				syn[s==0]=0
				resp=np.stack((ave,syn), axis=0) #shape=(type, layer, batch, channel)
				batchResponse.append(resp)
			batchResponse=np.concatenate(batchResponse, axis=-2)

			batchResponse=batchResponse.mean(axis=-2) #shape=(type, layer, channel)
			freqResponse.append(batchResponse)
		freqResponse=np.array(freqResponse) #shape=(freq, type, layer, channel)

		return freqResponse


def compNoiseAm0AveSyn(stimSec, waveFs, fileModel, architecture, gpu_id, trainingRms):
	with chainer.using_config("enable_backprop", False):
		if cupy is not None and gpu_id>=0:
			xp=cupy
			cupy.cuda.Device(gpu_id).use()
		else: xp=np

		print("Conducting physiology:", "stimulus AM freq:", 0)

		waveLen=int(stimSec*waveFs)
		np.random.seed(1)

		times=np.arange(waveLen)/waveFs
		freqs=np.logspace(np.log10(1), np.log10(2000), 2**8)
		meanSize=2**2  #we used 2**4 in our paper

		batchSizeUpper=meanSize
		batchSize=np.array_split(np.arange(meanSize), int(np.ceil(meanSize/batchSizeUpper)))
		batchSize=[len(x) for x in batchSize]
		segmentSecUpper=2 #decrease this for a GPU with smaller memory
		segmentLenUpper=int(segmentSecUpper*waveFs)

		net=loadNet(architecture, fileModel)
		if gpu_id>=0: net.to_gpu(gpu_id)
		inputLen=totalInputLength(net.structure)

		batchAve=[]
		batchSyn=[]
		for bi,bs in enumerate(batchSize):
			waves=np.random.randn(bs, waveLen)
			waves=scaleRms(waves, trainingRms)
# 			repre=compLongRepresentation(net, waves, segmentLenUpper, waveLen, xp) #shape=(layer, batch, channel, length) #compLongRepresentationSingle() returns the same result when trimInputLen=True
			repre=compLongRepresentationSingle(net, waves, xp)

			repre+=1 #elu
			ave=repre.mean(axis=-1) #shape=(layer, batch, channel)
			batchAve.append(ave)

			freqResponse=[]
			for fi, freq in enumerate(freqs):
				cos,sin=vectorCosSin(freq, inputLen, waveLen, waveFs)

				s=repre.sum(axis=-1)
				syn=(((repre*cos).sum(axis=-1)/s)**2+((repre*sin).sum(axis=-1)/s)**2)**0.5
				syn[s==0]=0 #shape=(layer, batch, channel)
				freqResponse.append(syn)
			freqResponse=np.array(freqResponse) #shape=(freq, layer, batch, channel)
			batchSyn.append(freqResponse)

		batchAve=np.concatenate(batchAve, axis=1) #shape=(layer, batch, channel)
		batchAve=batchAve.mean(axis=1) #shape=(layer, channel)

		batchSyn=np.concatenate(batchSyn, axis=2) #shape=(freq, layer, batch, channel)
		batchSyn=batchSyn.mean(axis=2) #shape=(freq, layer, channel)

		return batchAve, batchSyn

