import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class Node:
   def __init__(self):
       self.col = None
       self.children = {}
       self.output = None

def readData(path):
    with open(path,"r") as f:
        readFile = f.read()
        X = np.empty(0)
        y = np.empty(0)
        rows = 0
        cols = len(readFile.split("\n")[0].split("\t")) - 1
        for line in readFile.split("\n")[1:]:
            if line!="":
                X = np.append(X, np.array(line.split("\t")[:-1]))
                y = np.append(y, np.array(int(line.split("\t")[-1])))
                rows = rows + 1
        X = np.reshape(X, (rows, cols))
        y = np.reshape(y, (rows,1))        
        return X, y

def calculateStats(val):
    mean = val.sum()/val.shape[0]
    diff = ((val - mean)**2).sum()
    return math.sqrt(diff/(val.shape[0])),mean 

def partitiondata(col, labels, X, y):
    xDict = {}
    yDict = {}
    nodeDict = {}
    myCol = X[:, col]
    X = np.delete(X, col, axis=1)

    for label in labels:
        xDict[label] = np.empty((0, X.shape[1]))
        yDict[label] = np.empty((0,1))
        nodeDict[label] = Node()
    if X.shape[1]!=0:
        for i in range(X.shape[0]):
            xDict[myCol[i]] = np.vstack((xDict[myCol[i]], X[i,:]))
            yDict[myCol[i]] = np.vstack((yDict[myCol[i]], y[i,:]))
    
    else:
        for i in range(X.shape[0]):
            yDict[myCol[i]] = np.append(yDict[myCol[i]], y[i,:])          
    
    return xDict, yDict, nodeDict    
        
    

def findBestSplit(X, y, initial):
    maxGain = float("-inf")
    maxColGain = 0
    bestSplitCols = set()
    for i in range(X.shape[1]):
        overallStd = 0
        cols = set()
        for item in X[:,i]:
            cols.add(item)
        for val in cols:
            row_count = 0 
            y_temp = np.empty(0)
            for j in range(X.shape[0]):
                if X[j, i] == val:
                    y_temp = np.append(y_temp, y[j])
                    row_count = row_count + 1
            y_temp = np.reshape(y_temp, (row_count, 1))
            stddev ,_  = calculateStats(y_temp)
            overallStd += stddev*(row_count/X.shape[0]) 
        if initial - overallStd > maxGain :
            maxGain = initial - overallStd
            maxColGain = i
            bestSplitCols = cols
    return maxColGain, bestSplitCols       
    

def constructTree(X, y, N, flag):
    initial, mean = calculateStats(y)
    if initial/mean < 0.01 or flag == True:
        N.output = mean
        return    
    col, categories = findBestSplit(X, y, initial)
    N.col = col
    xDict, yDict, nodeDict = partitiondata(col, categories, X, y)
    N.children = nodeDict
    nextFlag = False
    if X.shape[1] == 1:
        nextFlag = True        
    for node in nodeDict:
        #print(node,nodeDict)
        constructTree(xDict[node], yDict[node], nodeDict[node], nextFlag)

def predict(N, data):
    n = N
    while(n.output==None):
        n = n.children[data[n.col]]
    print(n.output)    

if __name__=="__main__" :
    X, y = readData("ggData.csv")
    n = Node()
    constructTree(X, y, n, False)
    data = ['A','D','G']
    predict(n, data)
