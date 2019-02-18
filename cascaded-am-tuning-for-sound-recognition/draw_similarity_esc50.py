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

from physiology.mtf_analysis import compLayerMeasure, compRegionLayerSimilarity
from physiology.am_meta_analysis import readCumulative



if __name__=="__main__":
	dirRoot=Path("./cascaded-am-tuning-for-sound-recognition")
	
	dirCumulatives=dirRoot/"am-meta-analysis"/"cumulative"
	dirEsc=dirRoot/"ESC50"
	dirResult=dirEsc/"Results"/"Result11"
	dirFigure=dirResult/"Figure"
	
	dirResponse=dirResult/"Response"
	fileResponse=dirResponse/"Am"
	with open(fileResponse, "rb") as f: response=pickle.load(f)
	fileResponse=dirResponse/"Am0"
	with open(fileResponse, "rb") as f: response0Rate,response0Temp=pickle.load(f)
	
	regions=("AN","CN","SOC","NLL", "IC", "MGB","cortex")
	
	numLayer=response.shape[-2]
	numChannel=response.shape[-1]

	layerMeasures=compLayerMeasure(response, response0Rate)
	cumulatives=readCumulative([file for file in layerMeasures.keys() if "-" not in file], str(dirCumulatives))
	similarity=compRegionLayerSimilarity(layerMeasures, cumulatives, numLayer, regions, numChannel)
	
	plt.pcolormesh(similarity, cmap="inferno")
	plt.yticks(np.arange(len(regions))+0.5,regions)
	plt.xticks(np.arange(numLayer)+0.5, np.arange(numLayer)+1)
	plt.ylabel("Brain regions")
	plt.xlabel("DNN layers")
	plt.show()
	