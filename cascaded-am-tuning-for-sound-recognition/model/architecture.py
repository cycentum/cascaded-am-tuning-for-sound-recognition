###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Takuya Koumura, Hiroki Terashima, Shigeto Furukawa. "Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition". bioRxiv. Cold Spring Harbor Laboratory; (2018): 308999.
###


import numpy as np


def parseArchitecture(string, delimLayer=",", delimParam=":"):
	string=string.split(delimLayer)
	architecture=[]
	for s in string: architecture.append(tuple(map(int, s.split(delimParam))))
	return architecture


def readArchitecture(file):
	with open(file, "r") as f: text=list(f)
	architecture=parseArchitecture(text[0].strip())
	return architecture


def architectureStr(architecture, delimLayer=",", delimParam=":"):
	return delimLayer.join([delimParam.join(tuple(map(str,st))) for st in architecture])


def sampleArchitecture(numLayer, totalInputLenUpper, numChannel, filterLenUpper):
	architecture=[]
	
	inputLens=1-np.random.rand(numLayer)
	inputLens/=inputLens.sum()
	inputLens=np.random.multinomial(totalInputLenUpper-1, inputLens)+1
	
	for li in range(numLayer):
		inputLen=inputLens[li]
		filterLen=np.random.randint(min(2, inputLen), min(filterLenUpper, inputLen)+1)
		if filterLen==1: inputLen=1
		if filterLen>1 and (inputLen-1)%(filterLen-1)!=0:
			dil=(inputLen-1)//(filterLen-1)
			inputLen=dil*(filterLen-1)+1
		architecture.append((numChannel, inputLen, filterLen))
	
	return architecture
