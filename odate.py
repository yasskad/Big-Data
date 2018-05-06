import os
import numpy as np
from math import sqrt
from operator import add
import re
import utils


def find_outliers(sequence):
	def stripdate(x):
		x = utils.isdate(x, value=True)
		return x
	sequence = sequence.map(stripdate)
	sequence = sequence.map(lambda x: (x,1)).reduceByKey(add)
	
	def lessfrequent(x):
		if x[1] < 10:
			return True
		return False
	sequenceout = sequence.filter(lessfrequent).map(lambda x: x[0]) 
	return sequenceout.collect()

