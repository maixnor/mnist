
# script to find best results of mnist evaluation based on accuracy, same accuracy is not taken care of
import numpy as np

# opening file
with open('log.txt') as file:
    # reading all lines from file, skipping first 2 lines
    lines = file.readlines()[2:]
    # closing file
    file.close()

    # declaring list scores
    scores = []

    # appending characters until first space (accuracy) to scores list
    [scores.append(line.split(' ')[0]) for line in lines]
    
    # finding index of highest value using numpy
    idx = np.argmax(np.array(scores))    

    # printing entire line of best accuracy
    print(lines[idx])
