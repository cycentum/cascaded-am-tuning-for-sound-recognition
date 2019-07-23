###
# (c) 2019 Takuya KOUMURA.
#
# This is a part of the codes for the following paper:
# Koumura T, Terashima H, Furukawa S (2019) Cascaded Tuning to Amplitude Modulation for Natural Sound Recognition. J Neurosci 39(28):5517–5533.
###

import csv
import time
from matplotlib import pyplot as plt

def readTable(file, delimiter="\t"):
	with open(file, "r") as f:
		r=csv.reader(f, delimiter=delimiter)
		table=[]
		for line in r:
			row=[]
			table.append(row)
			for c in line:
				row.append(c)
	return table



def defaultColors(index=None, alpha=1):
	'''
	@param alpha: [0,1]
	'''
	colorStr = plt.rcParams['axes.prop_cycle'].by_key()['color']
	if index is None:
		colors=[]
		for cs in colorStr:
			r=int(cs[1:3],16)/255
			g=int(cs[3:5],16)/255
			b=int(cs[5:7],16)/255
			colors.append((r,g,b,alpha))
		return colors

	else:
		cs=colorStr[index]
		r=int(cs[1:3],16)/255
		g=int(cs[3:5],16)/255
		b=int(cs[5:7],16)/255
		return(r,g,b,alpha)


def localTimeStr(delim=""):
	lt=time.localtime()
	lts=(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec)
	length=(4,2,2,2,2,2)
	return delim.join([("{:0"+str(l)+"d}").format(s) for s,l in zip(lts, length)])
