###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Koumura T, Terashima H, Furukawa S (2019) Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition. J Neurosci 39(28):5517–5533.
###


import numpy as np
from numpy import int32
import scipy.stats


def compFilterType(response, threshold=0.8, thresholdAbs=0.1):
	'''
	@return: 0: low, 1: high, 2: band, 3: flat
	'''
	numLayer=response.shape[-2]
	numChannel=response.shape[-1]
	filtType=-np.ones((2,numLayer,numChannel),int32) #low, high, band, flat
	for tyi,ty in enumerate(("ave","syn")):
		for li in range(numLayer):
			for ch in range(numChannel):
				r=response[...,tyi,li,ch]
				ptp=np.nanmax(r)-np.nanmin(r)
				if np.isnan(r).all() or ptp<thresholdAbs:
					filtType[tyi,li,ch]=3
					continue
				peak=np.nanargmax(r)
				ma=np.nanmax(r)
				mi=0
				loCutoff=(r[:peak]<ma*threshold+mi*(1-threshold)).sum()>0
				hiCutoff=(r[peak:]<ma*threshold+mi*(1-threshold)).sum()>0
				if loCutoff:
					if hiCutoff: filt=2
					else: filt=1
				else:
					if hiCutoff: filt=0
					else: filt=3
				filtType[tyi,li,ch]=filt
	return filtType


def bmf(response, freqs, addNan=False):
	numLayer=response.shape[-2]
	layerMeasure=[]
	for li in range(numLayer):
		r=response[:,li]
		measure=[]
		for ch in range(r.shape[1]):
			rch=r[:,ch]
			if not addNan:
				if np.isnan(rch).all(): continue
				if (rch[~np.isnan(rch)]==rch[~np.isnan(rch)][0]).all(): continue
			elif np.isnan(rch).all() or (rch[~np.isnan(rch)]==rch[~np.isnan(rch)][0]).all():
				measure.append(np.NAN)
				continue
			meas=np.where(rch==np.nanmax(rch))[0]
			meas=freqs[meas]
			if len(meas)>1: meas=scipy.stats.gmean(meas)
			measure.append(float(meas))
		measure=np.array(measure)
		layerMeasure.append(measure)
	return layerMeasure


def cutoff(response, freqs, thresholdFunc, addNan=False):
	numLayer=response.shape[-2]
	layerMeasure=[]
	for li in range(numLayer):
		r=response[:,li]
		measure=[]
		for ch in range(r.shape[1]):
			rch=r[:,ch]
			if not addNan:
				if np.isnan(rch).all(): continue
				if (rch[~np.isnan(rch)]==rch[~np.isnan(rch)][0]).all(): continue
			elif np.isnan(rch).all() or (rch[~np.isnan(rch)]==rch[~np.isnan(rch)][0]).all():
				measure.append(np.NAN)
				continue
			peak=np.where(rch==np.nanmax(rch))[0][-1]
			th=thresholdFunc(rch)
			meas=np.where((rch[:-1]>th)*(rch[1:]<=th))[0]
			meas=meas[meas>=peak]
			if len(meas)==0:
				if addNan: measure.append(np.NaN)
				continue
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
