###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Takuya Koumura, Hiroki Terashima, Shigeto Furukawa. "Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition". bioRxiv. Cold Spring Harbor Laboratory; (2018): 308999.
###


import numpy as np
from numpy import newaxis, float32, float64, int32, int64, int16, int8, uint8, uint32
import scipy.stats
from collections import defaultdict
import itertools


def bmf(response, freqs, filtType=None):
	numLayer=response.shape[-2]
	layerMeasure=[]
	for li in range(numLayer):
		r=response[:,li]
		measure=[]
		for ch in range(r.shape[1]):
			rch=r[:,ch]
			if filtType is not None and (filtType[li,ch]==0 or filtType[li,ch]==3 or filtType[li,ch]==1): continue
			if np.isnan(rch).all(): continue
			if (rch[~np.isnan(rch)]==rch[~np.isnan(rch)][0]).all(): continue
			meas=np.where(rch==np.nanmax(rch))[0]
			meas=freqs[meas]
			if len(meas)>1: meas=scipy.stats.gmean(meas)
			measure.append(float(meas))
		measure=np.array(measure)
		layerMeasure.append(measure)
	return layerMeasure
	
	
def cutoff(response, freqs, thresholdFunc, filtType=None):
	numLayer=response.shape[-2]
	layerMeasure=[]
	for li in range(numLayer):
		r=response[:,li]
		measure=[]
		for ch in range(r.shape[1]):
			rch=r[:,ch]
			if filtType is not None and (filtType[li,ch]==1 or filtType[li,ch]==3): continue
			if np.isnan(rch).all(): continue
			if (rch[~np.isnan(rch)]==rch[~np.isnan(rch)][0]).all(): continue
			peak=np.nanargmax(rch)
			th=thresholdFunc(rch)
			meas=np.where((rch[:-1]>th)*(rch[1:]<=th))[0]
			meas=meas[meas>=peak]
			if len(meas)==0: continue
			measVal=np.empty(len(meas))
			for mi,m in enumerate(meas):
				f0,f1=np.log10(freqs[m:m+2])
				r0,r1=rch[m:m+2]
				measVal[mi]=(f0*r1-f1*r0+th*(f1-f0))/(r1-r0)
			meas=10**(np.array(measVal))
			if len(meas)>1: meas=scipy.stats.gmean(meas)
			measure.append(float(meas))
		measure=np.array(measure)
		layerMeasure.append(measure)
	return layerMeasure


def highestSignificant(response, freqs):
	'''
	tMTF cutoff freq = max freq with significant R
	highest freq with significant sync
	max freq with significant R
	'''
	numLayer=response.shape[-2]
	layerMeasure=[]
	for li in range(numLayer):
		r=response[:,li]
		measure=[]
		for ch in range(r.shape[1]):
			rch=r[:,ch]
			if np.isnan(rch).all(): continue
			meas=np.where(~np.isnan(rch))[0][-1]
			meas=freqs[meas]
			measure.append(float(meas))
		measure=np.array(measure)
		layerMeasure.append(measure)
	return layerMeasure


def compLayerMeasure(responseAmAveSyn, responseAm0Ave):
	freqs=np.logspace(np.log10(1), np.log10(2000), 2**8)

	layerMeasures={}

	response=responseAmAveSyn
	response[np.isnan(response)]=0
	numLayer=response.shape[-2]
	numChannel=response.shape[-1]
	responseAve=response[:,0]
	responseSyn=response[:,1]
	
	response0Ave=responseAm0Ave
	assert response0Ave.shape[-2]==numLayer
	assert response0Ave.shape[-1]==numChannel
	
	responseSyn[responseSyn<0.01]=np.NaN
	responseAve[responseAve<response0Ave+0.01]=np.NaN
	
	filtType=-np.ones((3,numLayer,numChannel),int32) #low, high, band, flat
	for tyi,ty in enumerate(("ave","syn","avesyn")):
		if ty=="ave": response=responseAve
		elif ty=="syn": response=responseSyn
		elif ty=="avesyn": response=responseAve*responseSyn
		for li in range(numLayer):
			for ch in range(numChannel):
				r=response[...,li,ch]
				ptp=np.nanmax(r)-np.nanmin(r)
				if np.isnan(r).all() or ptp<0.1:
					filtType[tyi,li,ch]=3
					continue
				peak=np.nanargmax(r)
				ma=np.nanmax(r)
				mi=0
				loCutoff=(r[:peak]<ma*0.8+mi*0.2).sum()>0
				hiCutoff=(r[peak:]<ma*0.8+mi*0.2).sum()>0
				if loCutoff:
					if hiCutoff: filt=2
					else: filt=1
				else:
					if hiCutoff: filt=0
					else: filt=3
				filtType[tyi,li,ch]=filt
	
	layerMeasure=bmf(responseAve, freqs, filtType[0])
	layerMeasures["Yin2011_cortex_st0_me1"]=layerMeasure
	layerMeasures["Preuss1990_MGB_st0_me0"]=layerMeasure
	layerMeasures["Langner1988_IC_st0_me0"]=layerMeasure
	layerMeasures["Schreiner1988_cortex_st0_me1"]=layerMeasure
	layerMeasures["Zhang2006_NLL_st0_me0"]=layerMeasure
	layerMeasures["Krishna2000_IC_st0_me2"]=layerMeasure
	layerMeasures["Bieser1996_cortex_st0_me1"]=layerMeasure
	layerMeasures["Batra2006_NLL_st0_me0"]=layerMeasure
	layerMeasures["Condon1996_IC_st0_me0"]=layerMeasure
	layerMeasures["Liang2002_cortex_st0_me0"]=layerMeasure
	layerMeasures["ave-bmf"]=layerMeasure

	layerMeasure=cutoff(responseAve, freqs, lambda rch: np.nanmin(rch)*0.9+np.nanmax(rch)*0.1, filtType[0])
	layerMeasures["Krishna2000_IC_st0_me3"]=layerMeasure
	
	layerMeasure=cutoff(responseAve, freqs, lambda rch: np.nanmax(rch)*0.8, filtType[0])
	layerMeasures["Kuwada1999_SOC_st0_me1"]=layerMeasure
	layerMeasures["ave-cutoff"]=layerMeasure
	
	layerMeasure=cutoff(responseAve, freqs, lambda rch: np.nanmax(rch)*0.5, filtType[0])
	layerMeasures["Zhang2006_NLL_st0_me1"]=layerMeasure
	
	r=responseAve.copy()
	layerMeasure=highestSignificant(r, freqs)
	layerMeasures["Lu2000_cortex_st0_me0"]=layerMeasure
	layerMeasures["Bartlett2007_MGB_st0_me0"]=layerMeasure
	layerMeasures["Lu2001_MGB_st0_me0"]=layerMeasure
	
	layerMeasure=bmf(responseSyn, freqs, filtType[1])
	layerMeasures["Preuss1990_MGB_st0_me1"]=layerMeasure
	layerMeasures["Frisina1990_CN_st0_me0"]=layerMeasure
	layerMeasures["Rhode1994_CN_st0_me1"]=layerMeasure
	layerMeasures["Zhang2006_NLL_st0_me2"]=layerMeasure
	layerMeasures["Kuwada1999_SOC_st0_me0"]=layerMeasure
	layerMeasures["Huffman1998_NLL_st0_me1"]=layerMeasure
	layerMeasures["Liang2002_cortex_st0_me1"]=layerMeasure
	layerMeasures["Schulze1997_cortex_st0_me1"]=layerMeasure
	layerMeasures["syn-bmf"]=layerMeasure

	layerMeasure=highestSignificant(responseSyn, freqs)
	layerMeasures["Lu2000_cortex_st0_me1"]=layerMeasure
	layerMeasures["Bartlett2007_MGB_st0_me1"]=layerMeasure
	layerMeasures["Lu2001_MGB_st0_me1"]=layerMeasure
	layerMeasures["Scott2011_cortex_st0_me2"]=layerMeasure
	layerMeasures["Batra1989_IC_st0_me1"]=layerMeasure
	layerMeasures["Kuwada1999_SOC_st0_me2"]=layerMeasure
	layerMeasures["Krishna2000_IC_st0_me4"]=layerMeasure
	layerMeasures["Batra2006_NLL_st0_me1"]=layerMeasure
	layerMeasures["Liang2002_cortex_st0_me2"]=layerMeasure
	
	layerMeasure=cutoff(responseSyn, freqs, lambda rch: np.nanmax(rch)*10**(-3/20), filtType[1])
	layerMeasures["Joris1992_AN_st0_me0"]=layerMeasure
	layerMeasures["JorisSmith1998_CN_st0_me0"]=layerMeasure
	layerMeasures["JorisYin1998_CN_st0_me0"]=layerMeasure
	layerMeasures["JorisYin1998_SOC_st0_me0"]=layerMeasure
	
	layerMeasure=cutoff(responseSyn, freqs, lambda rch: np.nanmax(rch)*0.8, filtType[1])
	layerMeasures["Rhode1994_AN_st0_me0"]=layerMeasure
	layerMeasures["Rhode1994_CN_st0_me0"]=layerMeasure
	layerMeasures["syn-cutoff"]=layerMeasure

	layerMeasure=cutoff(responseSyn, freqs, lambda rch: 0.1, filtType[1])
	layerMeasures["Zhao1995_CN_st0_me0"]=layerMeasure
	layerMeasures["Rhode1994_AN_st0_me2"]=layerMeasure
	layerMeasures["Rhode1994_CN_st0_me2"]=layerMeasure
	
	layerMeasure=bmf(responseAve*responseSyn, freqs, filtType[2])
	layerMeasures["Eggermont1998_cortex_st0_me0"]=layerMeasure
	layerMeasures["MullerPreuss1986_cortex_st0_me0"]=layerMeasure
	layerMeasures["MullerPreuss1986_IC_st0_me0"]=layerMeasure
	
	layerMeasure=cutoff(responseAve*responseSyn, freqs, lambda rch: np.nanmax(rch)*0.5, filtType[2])
	layerMeasures["Eggermont1998_cortex_st0_me1"]=layerMeasure
	
	return layerMeasures


def makePaperRateTemp():
	paperRateTemp=defaultdict(list)
	paperRateTemp[("ave","bmf")].append("Yin2011_cortex_st0_me1")
	paperRateTemp[("ave","bmf")].append("Preuss1990_MGB_st0_me0")
	paperRateTemp[("ave","bmf")].append("Langner1988_IC_st0_me0")
	paperRateTemp[("ave","bmf")].append("Schreiner1988_cortex_st0_me1")
	paperRateTemp[("ave","bmf")].append("Zhang2006_NLL_st0_me0")
	paperRateTemp[("ave","bmf")].append("Krishna2000_IC_st0_me2")
	paperRateTemp[("ave","bmf")].append("Bieser1996_cortex_st0_me1")
	paperRateTemp[("ave","bmf")].append("Batra2006_NLL_st0_me0")
	paperRateTemp[("ave","bmf")].append("Condon1996_IC_st0_me0")
	paperRateTemp[("ave","bmf")].append("Liang2002_cortex_st0_me0")

	paperRateTemp[("ave","cutoff")].append("Krishna2000_IC_st0_me3")
	paperRateTemp[("ave","cutoff")].append("Kuwada1999_SOC_st0_me1")
	paperRateTemp[("ave","cutoff")].append("Zhang2006_NLL_st0_me1")
	paperRateTemp[("ave","cutoff")].append("Lu2000_cortex_st0_me0")
	paperRateTemp[("ave","cutoff")].append("Bartlett2007_MGB_st0_me0")
	paperRateTemp[("ave","cutoff")].append("Lu2001_MGB_st0_me0")
	
	paperRateTemp[("syn","bmf")].append("Preuss1990_MGB_st0_me1")
	paperRateTemp[("syn","bmf")].append("Frisina1990_CN_st0_me0")
	paperRateTemp[("syn","bmf")].append("Rhode1994_CN_st0_me1")
	paperRateTemp[("syn","bmf")].append("Zhang2006_NLL_st0_me2")
	paperRateTemp[("syn","bmf")].append("Kuwada1999_SOC_st0_me0")
	paperRateTemp[("syn","bmf")].append("Huffman1998_NLL_st0_me1")
	paperRateTemp[("syn","bmf")].append("Liang2002_cortex_st0_me1")
	paperRateTemp[("syn","bmf")].append("Schulze1997_cortex_st0_me1")

	paperRateTemp[("syn","cutoff")].append("Lu2000_cortex_st0_me1")
	paperRateTemp[("syn","cutoff")].append("Bartlett2007_MGB_st0_me1")
	paperRateTemp[("syn","cutoff")].append("Lu2001_MGB_st0_me1")
	paperRateTemp[("syn","cutoff")].append("Scott2011_cortex_st0_me2")
	paperRateTemp[("syn","cutoff")].append("Batra1989_IC_st0_me1")
	paperRateTemp[("syn","cutoff")].append("Kuwada1999_SOC_st0_me2")
	paperRateTemp[("syn","cutoff")].append("Krishna2000_IC_st0_me4")
	paperRateTemp[("syn","cutoff")].append("Batra2006_NLL_st0_me1")
	paperRateTemp[("syn","cutoff")].append("Liang2002_cortex_st0_me2")
	paperRateTemp[("syn","cutoff")].append("Joris1992_AN_st0_me0")
	paperRateTemp[("syn","cutoff")].append("JorisSmith1998_CN_st0_me0")
	paperRateTemp[("syn","cutoff")].append("JorisYin1998_CN_st0_me0")
	paperRateTemp[("syn","cutoff")].append("JorisYin1998_SOC_st0_me0")
	paperRateTemp[("syn","cutoff")].append("Rhode1994_AN_st0_me0")
	paperRateTemp[("syn","cutoff")].append("Rhode1994_CN_st0_me0")
	paperRateTemp[("syn","cutoff")].append("Zhao1995_CN_st0_me0")
	paperRateTemp[("syn","cutoff")].append("Rhode1994_AN_st0_me2")
	paperRateTemp[("syn","cutoff")].append("Rhode1994_CN_st0_me2")
	
	paperRateTemp[("syn","bmf")].append("Eggermont1998_cortex_st0_me0")
	paperRateTemp[("syn","bmf")].append("MullerPreuss1986_cortex_st0_me0")
	paperRateTemp[("syn","bmf")].append("MullerPreuss1986_IC_st0_me0")
	
	paperRateTemp[("syn","cutoff")].append("Eggermont1998_cortex_st0_me1")
	return paperRateTemp


def compRegionLayerSimilarity(layerMeasures, cumulatives, numLayer, regions, numChannel):
	validNumLower=1
	
	layerDistance={}
	for file,layerMeasure in layerMeasures.items():
# 		print(file, [len(meas) for li,meas in enumerate(layerMeasure)])
		if "-" in file:
			layDis=np.array([float(len(meas)>0) for li,meas in enumerate(layerMeasure)])
			layerDistance[tuple(file.split("-"))]=layDis
			continue
		
		file=tuple(file.split("_"))
		cumulative=cumulatives[file]
		layDis=np.ones(numLayer)
		for li,meas in enumerate(layerMeasure):
			if len(meas)<validNumLower: continue
			
			meas=np.sort(meas)
			values=np.log10(meas)
			cum=np.linspace(0,1,len(values)+1)
			values=np.repeat(values, 2)
			cum=np.repeat(cum, 2)[1:-1]
			cum=np.stack((values,cum),axis=-1)
			values0=np.unique(cum[:,0])
			for v in values0:
				if (values==v).sum()<=2: continue
				index=np.where(values==v)[0]
				cum[index[1:-1],:]=np.NaN
			cum=np.stack((cum[~np.isnan(cum[:,0]),0],cum[~np.isnan(cum[:,1]),1]),axis=1)
			meas=cum
			
			values=np.sort(np.unique(np.concatenate((meas[:,0], cumulative[:,0]))))
			values=np.linspace(values.min(),values.max(),2**12)
			interpMeas=np.interp(values, meas[:,0], meas[:,1], 0, 1)
			interpCumulative=np.interp(values, cumulative[:,0], cumulative[:,1], 0, 1)
			distance=abs(interpMeas-interpCumulative).max()
			
			layDis[li]=distance
		layerDistance[tuple(file)]=layDis
	
	nameRateTemp=makePaperRateTemp()
	for key in nameRateTemp:
		for i in range(len(nameRateTemp[key])): nameRateTemp[key][i]=tuple(nameRateTemp[key][i].split("_"))
	
	distance=np.empty((2,2,len(regions),numLayer))
	for (ri,region),(rti,rateTemp),(bci,bestCut) in itertools.product(enumerate(regions),enumerate(("ave","syn")),enumerate(("bmf","cutoff"))):
		papers=sorted(set([name[0] for name in nameRateTemp[rateTemp,bestCut] if name[1]==region]))
		dists=[]
		for paper in papers:
			paperDist=[]
			for name in nameRateTemp[rateTemp,bestCut]:
				if name[1]!=region or name[0]!=paper: continue
				paperDist.append(layerDistance[name])
			assert len(paperDist)>0
			paperDist=np.array(paperDist).mean(axis=0)
			dists.append(paperDist)
		if len(dists)==0: distance[rti,bci,ri]=layerDistance[rateTemp,bestCut]
		else: distance[rti,bci,ri]=np.array(dists).mean(axis=0)
	similarity=1-distance.mean(axis=(0,1))
	return similarity