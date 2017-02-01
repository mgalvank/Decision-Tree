# Author: Mohit Galvankar
# Builds decision tree and classifies input data based on the tree generated

#b



from __future__ import division
import math
import operator
import csv
import random
import numpy as np
from collections import Counter

# import arff
import sys
# from tabulate import tabulate


class DNode:
    def __init__(self, feature=None, value=None,result=None,left = None,right = None):
        self.feature = feature
        self.value = value
        self.result = result
        self.left = left
        self.right = right


#Main fucntion
#Computes best information gain by iterating through all the features. Split data on the best info gain and creates tree nodes.
def calculate(data,userdepth,depth):
    label_index = getLabel()
    # print data
    #If the data is null return new node
    if len(data)== 0:
        return DNode()

    # check if the data has only one label class. If yes return that label class as decision tree result.
    if checkSame(data,label_index) == True:
        return DNode(result=results(data,label_index))

    if depth < userdepth:
        bgain = 0.0
        infogain = 0.0
        entrootdata = entropy_main(data)
        bfeature = 0
        bvalue = 0
        # print "entropy of root data node",entrootdata
        #iterate through data
        uniquedp =[]
        for feature in range (0,len(data[0])-1):
            #find unique data points
            # print "feature",feature
            uniquedatapoints = uniquedatapoints_fn(data,feature)
            # print "uniquedatapoints",uniquedatapoints
            #split data on each unique data point
            for uniquedatapoint in uniquedatapoints:
                # print "In calculate : uniquedatapoint for loop", uniquedatapoint
                # print "bgain in calculate",bgain
                # print "bgain in calculate for feature,dtapoint", (bgain, bfeature, bvalue)
                # print "infogain in calculate for datapoint and feature",(infogain,uniquedatapoint,feature)
                infogain, dleft, dright = splitdata(entrootdata, data, feature, uniquedatapoint)
                # if feature == bfeature and uniquedatapoint == bvalue:
                #
                # print "infogain in calculate and dleft,dright", (infogain, dleft, dright)
                if infogain > bgain:
                    # print "Inside if infogain >bgain in splitdata"
                    bfeature = feature
                    bvalue = uniquedatapoint
                    bgain = infogain
                    bdleft = dleft
                    bdright = dright
                    # print "bfeature,bvalues in splitdata", (bfeature, bvalue)

                # raw_input()
        if bgain>0 and len(bdleft)>0 and len(bdright)>0:
            # print "In if bgain>0 and len(dleft)>0 or len(dright)>0 in calculate"
            DNleft = calculate(bdleft,userdepth,depth+1)
            # print "DNleft" ,DNleft
            DNright = calculate(bdright,userdepth,depth+1)
            return DNode(feature=bfeature,value=bvalue,left=DNleft,right=DNright)

    return DNode(result = results(data,label_index))


def checkSame(data,traindata_label_index):
    a = []
    for i in data:
        a.append(i[traindata_label_index])
    if len(set(a))<=1:
        return True
    else: return False


def results(data,traindata_label_index):
    dict={}
    for row in data:
        dict.setdefault(row[traindata_label_index],0)
        dict[row[traindata_label_index]]+=1
    return max(dict.iteritems(), key=operator.itemgetter(1))[0]



#Calculate the unique data points in a column/feature
def uniquedatapoints_fn(data,feature):
    a = [] #append the data points in the feature column
    for i in data:
        a.append(i[feature])
    b = list(set(a)) #find list unique data points
    return b

def getLabel():
    return -1

#Calculate the distribution.how many 1's and 0's
def cal_distribution(p):
    count_pos = 0
    label_index = getLabel()
    for value in p:
        # print "value",value
        # raw_input()
        if int(value[label_index]) == 1:
            count_pos = count_pos + 1
    return count_pos


#calculate the probability
def cal_probability(pos,neg):
    probability = 0.0
    probability = pos/(pos+neg)
    # print "probability",probability
    return probability

#calculate the entropy
def entropy_main(p):
    pos = 0
    neg = 0
    pos = cal_distribution(p)
    neg = len(p) - pos

    # print "pos,neg distribution in entropy_main",(pos,neg)
    prob_pos = cal_probability(pos,neg)
    prob_neg = 1.0 - prob_pos
    if prob_pos == 1 or prob_neg == 1:
        return 1

    # print "prob_pos,neg in main" ,prob_pos,prob_neg
    ent = entropy(prob_neg,prob_pos)
    return ent

#calculate the actual entropy via formula
def entropy(prob_neg,prob_pos):
    ent = 0.0
    ent = -(prob_neg*math.log(prob_neg,2))-(prob_pos*math.log(prob_pos,2))
    return ent


def cal_infogain(entparentdata,eright,eleft,lendleft,lendright):
    infogain = entparentdata - (lendleft/(lendleft+lendright))*eleft - (lendright/(lendleft+lendright))*eright
    return infogain


def splitdata(entrootdata,data,feature,uniquedatapoint):
    # print "feature of split data", feature
    # print "Unique datapoint in splitdata",uniquedatapoint
    dleft =[]
    dright =[]


    for i in data:
        if i[feature] == uniquedatapoint:
            # print i[feature]
            dleft.append(i)
        else:
            dright.append(i)
    # print "dleft in splitdata",dleft
    # print "dright in splitdata",dright
    if len(dright)>0:
        entright =  entropy_main(dright)
    else: entright =0


    if len(dleft) > 0:
        entleft = entropy_main(dleft)
    else :entleft = 0

    infogain = cal_infogain(entrootdata,entright,entleft,len(dleft),len(dright))
    # print "infogain in splitdata",infogain
    return infogain,dleft,dright

    # print dleft
    # print dright

def printtree(tree,indent=''):
    if tree.result!=None:
        print "Result",str(tree.result)
    else:
        print "If Feature ",str(tree.feature)+' and Value '+str(tree.value)+" :"
        print(indent+'Tree left->')
        printtree(tree.left,indent + '  ')
        print(indent+'Tree right->')
        printtree(tree.right,indent + '  ')

def classify(tree,datapoint):

    if(tree.result != None):
        return tree.result

    feature = tree.feature
    value = tree.value
    if(value == datapoint[feature]):
        label=classify(tree.left,datapoint)
    else:label = classify(tree.right,datapoint)

    return label

def classify_accu(tree,tdata):
    count = 0
    label_index = getLabel()
    for i in tdata:
        predicted = classify(tree,i)
        # print "predicted for",i,"is",predicted
        solution = i[label_index]
        if int(predicted) == int(solution):
            count = count + 1
    accuracy = count/len(tdata)
    return accuracy


def compute_confmatrix(tree,tdata):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    n = len(tdata)
    label_index = getLabel()
    for i in tdata:
        predicted = classify(tree, i)
        # print "predicted ",predicted
        solution = i[label_index]
        # print "actual solution",solution
        # raw_input()
        if int(predicted) == 1 and int(solution) == 1:
            TP = TP + 1
        elif int(predicted) == 0 and int(solution)== 0:
            TN = TN + 1
        elif int(predicted) == 0 and int(solution) == 1:
            FN = FN + 1
        elif int(predicted) == 1 and int(solution) == 0:
            FP = FP + 1
    confusion_matrix = [[TN,FN],[FP,TP]]
    print confusion_matrix
    error = (FN+FP)/(n)
    print "Error ",error
    print "Confusion Matrix :"
    for i in confusion_matrix:
        print i
    # print tabulate([['Actual : No', TN, FP], ['Actual : Yes', FN,TP]], headers=[' N : %s' %(n),'Predicted : No', 'Predicted : Yes'],tablefmt='orgtbl')










def preprocess(f):
    datatemp = []
    data = []
    for i in f:
        i = i.rstrip('\n')
        i = i.lstrip(' ')
        datatemp.append(i.split(' '))
    for i in datatemp:
        del i[len(i) - 1]

    for i in datatemp:
        i = map(int, i)
        data.append(i)

    return data

def calculate_bootstrap_accu(b,testdata):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    k = []
    n = len(testdata)
    final = []
    label_index = getLabel()
    for j in range(0,len(b[0])):
        k[:] = []
        for i in b:
            k.append(i[j])
        # final.append(max(k))
        most_common, num_most_common = Counter(k).most_common(1)[0]
        final.append(most_common)

    count = 0
    solution1 = []
    for i in testdata:
        # print "predicted for",i,"is",predicted
        solution1.append(i[label_index])

    for i in range(0,len(final)):
        a = final[i]
        b = solution1[i]
        if int(a) == int(b):
            count = count + 1
        if int(a) == 1 and int(b) == 1:
            TP = TP + 1
        elif int(a) == 0 and int(b) == 0:
            TN = TN + 1
        elif int(a) == 0 and int(b) == 1:
            FN = FN + 1
        elif int(a) == 1 and int(b) == 0:
            FP = FP + 1
    confusion_matrix = [[TN, FN], [FP, TP]]
    print "Confusion Matrix :-"
    for i in confusion_matrix:
        print i
    # print tabulate([['Actual : No', TN, FP], ['Actual : Yes', FN, TP]],
    #                headers=[' N : %s' % (n), 'Predicted : No', 'Predicted : Yes'], tablefmt='orgtbl')
    accuracy1 = count/len(final)
    print "Accuracy :",accuracy1




def learn_bagged(tdepth, nummbags, datapath):
    solution =[]
    data_train = []
    data_test = []
    a = []
    b= []
    b[:] = []

    temptraindata = np.genfromtxt(datapath + "/agaricuslepiotatrain1.csv", delimiter=",", skip_header=1)
    traindata_label = temptraindata[:,20]
    traindata_temp = np.delete(temptraindata,[20,21] , axis = 1)
    traindata = np.append(traindata_temp, traindata_label[:, np.newaxis], axis=1)
    traindata = np.array(traindata).astype(int).tolist()

    temptestdata = np.genfromtxt(datapath + "/agaricuslepiotatest1.csv", delimiter=",", skip_header=1)
    testdata_label = temptestdata[:, 20]
    testdata_temp = np.delete(temptestdata,[20,21] , axis = 1)
    testdata = np.append(testdata_temp, testdata_label[:, np.newaxis], axis=1)
    testdata = np.array(testdata).astype(int).tolist()

    traindata_label_index = -1
    testdata_label_index  = -1

    sample_size = len(traindata)
    bootstrapdata = []

    for numbag in range(0,nummbags):
        tree = None
        a[:] = []
        temp = 0
        bootstrapdata[:] = []
        bootstrapdata = [random.choice(traindata) for _ in range(0,sample_size)]
        tree = calculate(bootstrapdata,tdepth,0)
        # printtree(tree)
        for datapoint in testdata:
            temp = classify(tree,datapoint)
            a.append(temp)
        # print a

        # print "For a  %s is %s"%(numbag,a)
        b.append(a)
        # raw_input()
        # print "For b %s is %s" %(numbag,b)

    # for i in b:
    #     print "For b",i


        confusion_matrix = []
        # compute_confmatrix(tree, testdata)

        # accuracy = classify_accu(tree, testdata)

        # print "Accuracy : ", accuracy
    print "Depth : %s | Bags : %s " %(tdepth,nummbags)
    print ""
    calculate_bootstrap_accu(b,testdata)




    # print "Value of k", k


# if __name__ == "__main__":
#     tdepth = 5
#     nummbags = 5
#     datapath = "C:/Users/Mohit/PycharmProjects/DecisionTree/mushrooms/"
#     learn_bagged(tdepth, nummbags, datapath)