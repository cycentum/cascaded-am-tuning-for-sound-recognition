###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Takuya Koumura, Hiroki Terashima, Shigeto Furukawa. "Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition". bioRxiv. Cold Spring Harbor Laboratory; (2018): 308999.
###

import chainer
from chainer import Variable, serializers, functions, links, optimizers, cuda, Chain
import numpy as np
from numpy import newaxis, float32, float64, int32, int64, int16, int8, uint8, uint32

try:
	import cupy
except:
	cupy=None
	pass

class Net(chainer.Chain):
	def __init__(self, numLabel, architecture, act):
		'''
		@param architecture: ((channel, input len, filter len), ...)
		'''
		super(Net, self).__init__()
		
		self.numLabel=numLabel
		self.structure=architecture
		self.act=act
		
		self.prev=[None]*len(architecture)
		
		for li,st in enumerate(architecture):
			numChannel,inputLen,filterLen=st
			
			if li==0: inChannel=1
			else: inChannel=architecture[li-1][0]
			
			if filterLen==1:
				assert inputLen==1
				dil=1
			else:
				assert (inputLen-1)%(filterLen-1)==0
				dil=(inputLen-1)//(filterLen-1)
			
			conv=links.DilatedConvolution2D(inChannel, numChannel, (filterLen,1), 1, 0, (dil,1))
			super(Net, self).add_link("c"+str(li), conv)
		
		if numLabel>0:
			full=links.Convolution2D(architecture[-1][0], numLabel, 1)
			super(Net, self).add_link("full", full)
		
	def __call__(self, x, train):
		'''
		@param x: shape=(batch, channel=1, length, 1)
		@return: shape=(batch, label, length, 1)
		'''
		if cupy is None: xp=np
		else: xp=cupy.get_array_module(x.data)
		numBatch=x.shape[0]
		for li,st in enumerate(self.structure):
			numChannel,inputLen,filterLen=st
			prevLen=inputLen-1
			p=self.prev[li]
			if x.shape[2]>prevLen: self.prev[li]=xp.split(x.data, (x.shape[2]-prevLen,), axis=2)[1]
			elif p is not None:
				self.prev[li]=xp.concatenate((p, x.data), axis=2)
				if self.prev[li].shape[2]>prevLen: self.prev[li]=xp.split(self.prev[li], (self.prev[li].shape[2]-prevLen,), axis=2)[1]
			else: self.prev[li]=x.data
			if p is None: p=xp.zeros((numBatch, 1 if li==0 else self.structure[li-1][0], prevLen, 1), float32)
			if p.shape[2]<prevLen: p=xp.concatenate((xp.zeros((p.shape[0], p.shape[1], prevLen-p.shape[2], p.shape[3]), float32), p), axis=2)
			p=Variable(p)
			x=functions.concat((p, x), axis=2)
			
			x=self["c"+str(li)](x)
			x=self.act(x)
		
		if self.numLabel==0: return x
		x=self.full(x)
		return x
	
	def callSingle(self, x, train):
		for li,st in enumerate(self.structure):
			x=self["c"+str(li)](x)
			x=self.act(x)
		
		if self.numLabel==0: return x
		x=self.full(x)
		return x
	
	def checkLength(self, x):
		numBatch=x.shape[0]
		for li,st in enumerate(self.structure):
			p=self.prev[li]
			if(x.shape[2]<=self.structure[li][1]-1): return False
			self.prev[li]=np.split(x.data, (x.shape[2]-(self.structure[li][1]-1), ), axis=2)[1]
			if p is None: p=np.zeros((numBatch, 1, self.structure[li][1]-1, 1), float32)
			x=np.concatenate((p,x),axis=2)
			
			numChannel,inputLen,filterLen=st
			if filterLen==1: dil=1
			else: dil=(inputLen-1)//(filterLen-1)
			newLength=(x.shape[2]-filterLen-(filterLen-1)*(dil-1))+1
			x=x[:,0,newaxis,:newLength,:]
		return True


	def reset(self):
		for li,p in enumerate(self.prev): self.prev[li]=None


def totalInputLength(architecture):
	return np.array([st[1]-1 for st in architecture]).sum()+1
