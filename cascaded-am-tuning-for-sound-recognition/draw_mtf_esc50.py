###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Takuya Koumura, Hiroki Terashima, Shigeto Furukawa. "Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition". bioRxiv. Cold Spring Harbor Laboratory; (2018): 308999.
###

import numpy as np
import pickle
from pathlib import Path
from matplotlib import pyplot as plt
from numpy import uint8
from PIL import Image


if __name__=="__main__":
	dirRoot=Path("./cascaded-am-tuning-for-sound-recognition")
	
	dirEsc=dirRoot/"ESC50"
	dirResult=dirEsc/"Results"/"Result11"
	dirFigure=dirResult/"Figure"
	
	dirResponse=dirResult/"Response"
	fileResponse=dirResponse/"Am"
	with open(fileResponse, "rb") as f: response=pickle.load(f)
	
	numLayer=response.shape[-2]
	numChannel=response.shape[-1]

	cm=plt.get_cmap("viridis")
	gray=np.array((127,127,127,255), uint8)
	for tyi,ty in enumerate(("rate","temp")):
		for li in range(numLayer):
			r=response[...,tyi,li,:]
			
			validIndex=(~(np.isnan(r)|(r==0))).any(axis=0)
			rValid=r[:,validIndex]
			rValid=rValid[:,np.nanargmax(rValid, axis=0).argsort()]
			
			rValid/=np.nanmax(rValid, axis=0)
			mi=0 
			ma=1
			
			r=np.concatenate((rValid, r[:,~validIndex]), axis=1)
			
			im=(cm((r.T-mi)/(ma-mi))*255).astype(uint8)
			im=im.reshape(r.size, im.shape[-1])
			im[np.isnan(r).reshape(r.size),:]=gray
			im=im.reshape(r.shape[1],r.shape[0],im.shape[-1])
			im=np.flipud(im)
			fileFig=dirFigure/"Mtf"/(ty+"_layer"+str(li)+".png")
			fileFig.parent.mkdir(exist_ok=True, parents=True)
			with open(fileFig, "wb") as f: Image.fromarray(im).save(fileFig)