import os
import re

def isfloat(x):
	try:
		float(x)
		return True
	except ValueError:
		return False

def isemail(x):
	pattern = ''
	return bool(re.match(pattern, x))

def isphonenumber(x):
	pattern = ''
	return bool(re.match(pattern, x))
	
def isdate(x, value=False):
	x = re.split('T| ',x)[0]
	delimiter = '([- /.:])'
	year = '((19|20)\d\d)'
	month = '(0[1-9]|1[012])'
	date = '(0[1-9]|[12][0-9]|3[01])'
	pattern1 = year + delimiter + month + delimiter + date
	pattern2 =  month + delimiter + date + delimiter + year
	pattern3 =  date + delimiter + month + delimiter + year	
	patternlist = [pattern1, pattern2, pattern3]
	for pattern in patternlist:
		if bool(re.match(pattern, x)):
			if value == False:
				return True
			else:
				if pattern == pattern1:
					return x[:4]
				else:
					return x[-4:]
	return False

def isstring(x):
	pattern = '^[a-zA-z .,;:]+$'
	return bool(re.match(pattern, x))

def gettype(seq):
	string = [isstring(x) for x in seq]
	if sum(string) > len(seq)*0.75: return 'String' 
	numeric = [isfloat(x) for x in seq]
	if sum(numeric) > len(seq)*0.75: return 'Numeric'
	#date = [isdate(x) for x in seq]
	#if sum(date) > len(seq)*0.75: return 'Date' 
	return 'None'
		

if __name__=='__main__':
	print(isfloat('23'))
	print(isfloat('23.432'))
	print(isfloat('af')) 
	
	print(isstring('dsas'))
	print(isstring('a fda'))
	print(isstring('ad@gmail.com'))
	print(isstring('23-10-1992'))

	print(gettype(['22','321','321','33', '4']))
	print(gettype(['22','djkas da','NY','dsa', 'a']))

	valuelist = ['22','djkas da','NY','dsa', 'a']
	valuelist = list(filter((lambda x: isfloat(x)), valuelist))
	print(valuelist)

	print(isdate('1992:12:09', value=True))
	print(isdate('2015-07-09T00:00:00', value=True))
