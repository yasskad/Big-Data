import os
import numpy as np
from math import sqrt
from operator import add
import re
import utils

def find_outliers(sequence):
	sequencenew = sequence.map(lambda x: (x,1)).reduceByKey(add)
	top3 = sorted(sequencenew.collect())[:3]
	a = top3[0]
	b = top3[1]
	c = top3[2]
	if float(a[1])/float(b[1]) > 20*float(b[1])/float(c[1]):
		return [a[0], a[1]]
	return []
	
