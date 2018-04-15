#!user/bin/env python
import os
import sys
import json
import pandas
import random

import operator
import datetime

from csv import reader
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import Row
from pyspark.sql import *
import utils
import onumeric
import ostring

def findoutliers(database):
	spark = SparkSession.builder.appName("Python Spark SQL basic example").config("spark.some.config.option", "some-value").getOrCreate()
	sc = spark.sparkContext
	sc.addFile("utils.py")
	sc.addFile("onumeric.py")
	sc.addFile("ostring.py")
	lines = sc.textFile(database)
	headers = lines.first()
	rows = lines
	#rows = lines.filter(lambda x: x!=headers)
	headers = headers.split("\t")
	rows = rows.map(lambda l: l.split("\t"))	
	rowsdf = spark.createDataFrame(rows)

	database = sc.parallelize([(headers[i], rowsdf.select("_"+str(i+1)).rdd.flatMap(lambda x: x).collect()) for i in sc.range(len(headers)).collect()])

	def detect(x):
		key, valuelist = x
		valuelist = valuelist[1:]
		rand_smpl = [valuelist[i] for i in sorted(random.sample(range(len(valuelist)), 10))]
		smpl_type = utils.gettype(rand_smpl)
		if smpl_type == 'Numeric':
			valuelist = list(filter((lambda x: utils.isfloat(x)), valuelist))
			valuelist = onumeric.find_outliers(sc, valuelist)
		elif smpl_type == 'String':
			valuelist = ostring.find_outliers(valuelist)
		elif smpl_type == 'None':
			valuelist = []
		return (key, valuelist) 
	
	database = database.map(detect)
	
	database.map(lambda x: "{0} \t {1}".format(x[0], x[1])).saveAsTextFile("x.out") 	

if __name__=='__main__':
	database = sys.argv[1]
	findoutliers(database)
#spark-submit --conf spark.pyspark.python=/share/apps/python/3.4.4/bin/python outlier.py /user/rtg267/zs4w-c9cd.tsv
