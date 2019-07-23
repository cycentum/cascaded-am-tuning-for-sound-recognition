###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Koumura T, Terashima H, Furukawa S (2019) Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition. J Neurosci 39(28):5517–5533.
###

import pickle
import soundfile
import os
from numpy import newaxis, float32, float64, int32, int64, int16, int8, uint8, uint32
import numpy as np

from utils.utils import readTable



def loadRawData(dirData):
	'''
	@param trueType: PHN or WRD or TXT
	@return infos: [(group name, dialect, speaker, sentence, wave len), ...]
	'''

	groupName=("TRAIN", "TEST")
	numDialect=8
	trueType="PHN"
	trueStride=1
	singleSpeaker=False
	returnLabelInterval=False

	infos=[]
	waves=[]
	trues=[]
	truesAll=[]
	dirData=str(dirData)
	for gn in groupName:
		for dialect in range(numDialect):
			speakers=sorted(os.listdir(dirData+"/"+gn+"/DR"+str(dialect+1)))
			for speaker in speakers:
				files=sorted(list(filter(lambda x: x[-4:]==".WAV", os.listdir(dirData+"/"+gn+"/DR"+str(dialect+1)+"/"+speaker))))
				for file in files:
					sentence=file[:-4]
					text=readTable(dirData+"/"+gn+"/DR"+str(dialect+1)+"/"+speaker+"/"+sentence+"."+trueType, " ")
					for line in text: assert len(line)==3, dirData+"/"+gn+"/DR"+str(dialect+1)+"/"+speaker+"/"+sentence+"."+trueType
					truesAll.append(text)

					if not singleSpeaker or dialect==0 and speaker==speakers[0]:
						wave,fs=soundfile.read(dirData+"/"+gn+"/DR"+str(dialect+1)+"/"+speaker+"/"+file, dtype=float32)
						waves.append(wave)
						infos.append((gn, dialect, speaker, sentence, len(wave)))
						trues.append(text)

# 	silentLabels={"h#", "pau"}
	silentLabels={"h#", }
	silentLabel="h#"
	labels=set()
	for tru in truesAll:
		for _,_,la in tru: labels.add(la.strip())
	labels=sorted(list(labels))
	for la in silentLabels: labels.remove(la)
	labels.append(silentLabel)
	labelIndex={}
	for li,la in enumerate(labels): labelIndex[la]=li

	if returnLabelInterval: labelInterval=[[] for i in range(len(infos))]
	for i,(tru,info) in enumerate(zip(trues,infos)):
		le=info[4]//trueStride
		trueArray=np.ones(le, int32)*labelIndex[silentLabel]
		for ti,t in enumerate(tru):
			t0=int(t[0])//trueStride
			t1=int(t[1])//trueStride
			la=t[2].strip()
			if la in silentLabels: la=silentLabel
			li=labelIndex[la]
			if returnLabelInterval: labelInterval[i].append(((t0,t1),li))
			trueArray[t0:t1]=li
			t[2]=la
		trues[i]=trueArray

	if returnLabelInterval: return infos, waves, trues, labels, labelInterval
	return infos, waves, trues, labels, fs


def collapseLabel39(trues, labels):
	labelGroup=(
		("aa", "ao"),
		("ah", "ax", "ax-h"),
		("er", "axr"),
		("hh", "hv"),
		("ih", "ix"),
		("l", "el"),
		("m", "em"),
		("n", "en", "nx"),
		("ng", "eng"),
		("sh", "zh"),
		("uw", "ux"),
		("pcl", "tcl", "kcl", "bcl", "dcl", "gcl", "h#", "pau", "epi"),
	)

	labelGroupMap={}
	for la in labels: labelGroupMap[la]=la
	for lg in labelGroup:
		for la in lg: labelGroupMap[la]=lg[0]

	newLabels=sorted(set([labelGroupMap[la] for la in labels]))
	newLabels.remove("q")

	newLabelIndex={}
	for li,la in enumerate(newLabels): newLabelIndex[la]=li
	newLabelIndex["q"]=-1

	for tri,tr in enumerate(trues):
		newTr=np.empty_like(tr)
		for li,la in enumerate(labels):
			newTr[tr==li]=newLabelIndex[labelGroupMap[la]]
			trues[tri]=newTr

	return newLabels


def loadData(dirData):
	infos, waves, trues, labels, fs=loadRawData(dirData)
	labels=collapseLabel39(trues, labels)
	return infos, waves, trues, labels, fs

