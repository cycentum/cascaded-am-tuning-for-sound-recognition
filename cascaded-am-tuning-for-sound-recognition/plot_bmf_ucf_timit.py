###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Koumura T, Terashima H, Furukawa S (2019) Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition. J Neurosci 39(28):5517–5533.
###

import numpy as np
import pickle
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from utils.utils import defaultColors
from physiology.mtf_plot import compFilterType, bmf, cutoff


if __name__=="__main__":
	dirRoot=Path("./cascaded-am-tuning-for-sound-recognition")

	dirTimit=dirRoot/"TIMIT"
	dirResult=dirTimit/"Results"/"Result22"
	dirFigure=dirResult/"Figure"

	dirResponse=dirResult/"Response"
	fileResponse=dirResponse/"Am"
	with open(fileResponse, "rb") as f: response=pickle.load(f)
	fileResponse=dirResponse/"Am0"
	with open(fileResponse, "rb") as f: response0Ave, response0Syn=pickle.load(f)

	freqs=np.logspace(np.log10(1), np.log10(2000), 2**8)
	numLayer=response.shape[-2]
	numChannel=response.shape[-1]
	assert response0Ave.shape[-2]==numLayer
	assert response0Ave.shape[-1]==numChannel

	response[:,1][response[:,1]<0.01]=np.NaN
	response[:,0][response[:,0]<response0Ave+0.01]=np.NaN
	filtType=compFilterType(response)

	def calcHist(f):
		lf=np.log10(f)
		numBin=np.ceil(lf.ptp()/binWidth)
		m=(lf.min()+lf.max())/2
		bi=np.linspace(m-binWidth*numBin/2,m+binWidth*numBin/2, numBin+1)
		hist,bi=np.histogram(lf, bi)
		return hist,bi

	binWidth=0.2
	histograms={}
	for tyi,ty in enumerate(("ave","syn")):
		bestFreqs=bmf(response[...,tyi,:,:], freqs, True)
		cutoffs=cutoff(response[...,tyi,:,:], freqs, lambda rch: np.nanmax(rch)*0.8, True)
		for li in range(numLayer):
			f=bestFreqs[li]
			f[np.logical_or(filtType[tyi,li]==0,filtType[tyi,li]==3,filtType[tyi,li]==1)]=np.NaN
			f=f[~np.isnan(f)]
			if len(f)>0 and f.ptp()>0: histograms[tyi,0,li]=calcHist(f)
			sumBest=len(f)
			f=cutoffs[li]
			f[np.logical_or(filtType[tyi,li]==1,filtType[tyi,li]==3)]=np.NaN
			f=f[~np.isnan(f)]
			if len(f)>0 and f.ptp()>0: histograms[tyi,1,li]=calcHist(f)
			sumCut=len(f)
# 			print(ty,li,sumBest,sumCut,sep="\t")

	grid=GridSpec(numLayer,2)

# 	if len([h.max() for h,b in histograms.values()])==0: return
	ma=max([h.max() for h,b in histograms.values()])
	print("hist max", ma)
	ma=58
	binMin=min([b.min() for h,b in histograms.values()])
	binMax=max([b.max() for h,b in histograms.values()])
	print("bin min", binMin, "bin max", binMax)
	binMin=-0.0365397864987
	binMax=3.39000592068
	plt.figure(figsize=(4,14))
	for tyi,ty in enumerate(("rate","temp")):
		for li in range(numLayer):
# 			sp=plt.subplot(grid[numLayer-1-li,tyi])
			sp=plt.subplot(grid[numLayer-1-li,1-tyi])
			sp.set_title(ty+" "+str(li))
			if (tyi,1,li) in histograms:
				hist,bins=histograms[tyi,1,li]
				sp.bar((bins[1:]+bins[:-1])/2, hist, np.diff(bins), edgecolor=defaultColors(1), facecolor=defaultColors(1,0.3))
			if (tyi,0,li) in histograms:
				hist,bins=histograms[tyi,0,li]
				sp.bar((bins[1:]+bins[:-1])/2, hist, np.diff(bins), edgecolor=defaultColors(0), facecolor=defaultColors(0,0.3))
			sp.set_xlim(binMin-0.1,binMax+0.1)
			sp.set_ylim(0,ma*1.05)
			sp.set_xticks((0,1,2,3))
			sp.tick_params(color=(0.5,0.5,0.5), which="both", direction="in", labelcolor=(0.5,0.5,0.5), width=1)
			for spine in sp.spines: sp.spines[spine].set_color((0.5,0.5,0.5))
	plt.tight_layout(pad=0.5)
	plt.show()
# 	plt.savefig(fileFig, format="eps")
