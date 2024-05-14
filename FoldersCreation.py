# Importing the Libraries Required

import os
import string

# Creating the directory Structure

if not os.path.exists("dataSet_honey"):
    os.makedirs("dataSet_honey")

if not os.path.exists("dataSet_honey/trainingData"):
    os.makedirs("dataSet_honey/trainingData")

if not os.path.exists("dataSet_honey/testingData"):
    os.makedirs("dataSet_honey/testingData")

if not os.path.exists("dataSet_honey/validationData"):
    os.makedirs("dataSet_honey/validationData")

# Making folder  0 (i.e blank) in the training and testing data folders respectively
for i in range(0):
    if not os.path.exists("dataSet_honey/trainingData/" + str(i)):
        os.makedirs("dataSet_honey/trainingData/" + str(i))

    if not os.path.exists("dataSet_honey/testingData/" + str(i)):
        os.makedirs("dataSet_honey/testingData/" + str(i))

    if not os.path.exists("dataSet_honey/validationData/" + str(i)):
        os.makedirs("dataSet_honey/validationData/" + str(i))

# Making Folders from A to Z in the training and testing data folders respectively

for i in string.ascii_uppercase:
    if not os.path.exists("dataSet_honey/trainingData/" + i):
        os.makedirs("dataSet_honey/trainingData/" + i)
    
    if not os.path.exists("dataSet_honey/testingData/" + i):
        os.makedirs("dataSet_honey/testingData/" + i)

    if not os.path.exists("dataSet_honey/validationData/" + i):
        os.makedirs("dataSet_honey/validationData/" + i)

