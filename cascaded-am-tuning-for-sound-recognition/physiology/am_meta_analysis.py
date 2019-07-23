###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Koumura T, Terashima H, Furukawa S (2019) Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition. J Neurosci 39(28):5517–5533.
###


import numpy as np


def readCumulative(names, dirCum):
	cumulatives={}
	for file in names:
		file=tuple(file.split("_"))
		paper=file[0]
		cumulative=np.loadtxt(dirCum+"/"+paper+"/"+"_".join(file[1:])+".txt")
		assert cumulative[0,1]==0 and cumulative[-1,1]==1
		cumulative[cumulative[:,0]==0,0]=0.1
		cumulative[:,0]=np.log10(cumulative[:,0])
		cumulatives[file]=cumulative
	return cumulatives
