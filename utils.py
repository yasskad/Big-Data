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
	
def isdate(x):
	pattern = ''
	return bool(re.match(pattern, x))

def isstring(x):
	pattern = '^[a-zA-z .,;:]+$'
	return bool(re.match(pattern, x))

def gettype(seq):
	string = [isstring(x) for x in seq]
	if sum(string) > len(seq)*0.75: return 'String' 
	numeric = [isfloat(x) for x in seq]
	if sum(numeric) > len(seq)*0.75: return 'Numeric'
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
