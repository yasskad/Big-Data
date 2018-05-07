import os
import sys
#####
if __name__=='__main__':
        dataset = sys.argv[1]
        column = int(sys.argv[2])
        file = open(dataset, 'r')
        new_file = open('column.txt', 'w')
        for line in file:
                words = line.split('\t')
                new_file.write(words[column]+'\n')
        new_file.close()
