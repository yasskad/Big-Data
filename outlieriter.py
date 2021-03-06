#!user/bin/env python
import os
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
import odate
import onull

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

	outlierslist = []
	for i in sc.range(len(headers)).collect():
		header = headers[i]
		valuelistrdd = rowsdf.select("_"+str(i+1)).rdd.flatMap(lambda x: x)
		valuelist = valuelistrdd.collect()
		valuelist = valuelist[1:]
		rand_smpl = [valuelist[i] for i in sorted(random.sample(range(len(valuelist)), 10))]
		smpl_type = utils.gettype(rand_smpl)
		if smpl_type == 'Numeric':
			valuelistnew = onull.find_outliers(valuelistrdd)
			valuelistrdd = valuelistrdd.filter(lambda x: utils.isfloat(x))
			valuelistnew = valuelistnew + onumeric.find_outliers(valuelistrdd)
		elif smpl_type == 'String':
			valuelistnew = []
			valuelistnew = ostring.find_outliers(valuelistrdd)
		elif smpl_type == 'Date':
			valuelistnew = onull.find_outliers(valuelistrdd)
			valuelistrdd = valuelistrdd.filter(lambda x: utils.isdate(x))
			valuelistnew = valuelistnew + odate.find_outliers(valuelistrdd)
		elif smpl_type == 'None':
			valuelistnew = []
		outlierslist.append((header, valuelistnew))
	
	database = sc.parallelize(outlierslist)
	
	database.map(lambda x: "{0} \t {1}".format(x[0], x[1])).saveAsTextFile("x.out") 	

if __name__=='__main__':
	database = sys.argv[1]
	findoutliers(database)
#spark-submit --conf spark.pyspark.python=/share/apps/python/3.4.4/bin/python outlieriter.py /user/rtg267/zs4w-c9cd.tsv
