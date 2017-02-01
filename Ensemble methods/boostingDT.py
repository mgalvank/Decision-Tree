from __future__ import division

import math
import operator
import DataProcessing

class DNode:
    def __init__(self, feature=None, value=None, result=None, left=None, right=None):
        self.feature = feature
        self.value = value
        self.result = result
        self.left = left
        self.right = right

# Main fucntion
# Computes best information gain by iterating through all the features.
# Split data on the best info gain and creates tree nodes.
def calculate(data, userDepth, depth, weightList):
    # If the data is null return new noxde
    if len(data) == 0:
        return DNode()

    if len(set([x[0] for x in data])) <= 1:
        return DNode(result=results(data))

    if depth < userDepth:
        bestGain, infoGain, bestFeature, bestValue = 0.0, 0.0, 0, 0

        entropyData = entropy_main(data, weightList)

        for feature in range(1, len(data[0])):
            uniqueDataPoints = list(set([x[feature] for x in data]))
            for uniqueDataPoint in uniqueDataPoints:
                infoGain, dLeft, dRight, wleft, wright = splitdata(entropyData, data, feature, uniqueDataPoint, weightList)
                if round(infoGain, 10) > round(bestGain,10):
                    bestFeature, bestValue, bestGain, bestDLeftNode, bestDRightNode, bestWLeft, bestWRight \
                        = feature, uniqueDataPoint, infoGain, dLeft, dRight, wleft, wright

        if bestGain > 0 and len(bestDLeftNode) > 0 and len(bestDRightNode) > 0:
            dLeftNode = calculate(bestDLeftNode, userDepth, depth + 1, bestWLeft)
            dRightNode = calculate(bestDRightNode, userDepth, depth + 1, bestWRight)

            return DNode(feature=bestFeature, value=bestValue, left=dLeftNode, right=dRightNode)

    return DNode(result=results(data))

def results(data):
    dict = {}
    for row in data:
        dict.setdefault(row[0], 0)
        dict[row[0]] += 1
    return max(dict.iteritems(), key=operator.itemgetter(1))[0]

# calculate the entropy
def entropy_main(p, weights):
    pos = sum(weights[x] for x in range(len(p)) if p[x][0] == 1)
    neg = sum(weights[y] for y in range(len(p)) if p[y][0] == -1)

    prob_pos = float(pos / (pos + neg))
    prob_neg = 1.0 - prob_pos

    if prob_pos == 1:
        return 0
    if prob_neg == 1:
        return 0

    return -(prob_neg * math.log(prob_neg, 2)) - (prob_pos * math.log(prob_pos, 2))

# calculate infogain
def cal_infogain(entparentdata, eright, eleft, lendleft, lendright):
    infogain = entparentdata - (lendleft / (lendleft + lendright)) * eleft - (lendright / (
    lendleft + lendright)) * eright
    return infogain

# Splitting the data for a given node based on a given feature
def splitdata(entropy, data, feature, uniqueDataPoint, weightList):
    dleft, dright, weightLeft, weightRight, entright, entleft = [], [], [], [], 0, 0

    for i in range(len(data)):
        if data[i][feature] == uniqueDataPoint:
            dleft.append(data[i])
            weightLeft.append(weightList[i])
        else:
            dright.append(data[i])
            weightRight.append(weightList[i])

    if len(dright) > 0:
        entright = entropy_main(dright, weightRight)

    if len(dleft) > 0:
        entleft = entropy_main(dleft, weightLeft)

    infogain = cal_infogain(entropy, entright, entleft, len(dleft), len(dright))

    return infogain, dleft, dright, weightLeft, weightRight

# Classify the test point for a given tree and return the label
def classify(tree, datapoint):
    if (tree.result != None):
        return tree.result

    feature = tree.feature
    value = tree.value
    if value == datapoint[feature]:
        label = classify(tree.left, datapoint)
    else:
        label = classify(tree.right, datapoint)

    return label


def classify_accu(model, tdata):
    count, TN, TP, FN, FP = 0, 0, 0, 0, 0

    for i in tdata:
        value = 0.0
        for each in model.keys():
            value += model[each][1]*(classify(model[each][0], i))
        if value > 0:
            predicted = 1
        else:
            predicted = -1
        # print "predicted for",i,"is",predicted
        solution = i[0]
        if predicted == solution:
            count = count + 1

#code for confusion matrix, can be uncommented to check the results

    #     if predicted == 1 and solution == 1:
    #         TP = TP + 1
    #     elif predicted == -1 and solution == -1:
    #         TN = TN + 1
    #     elif predicted == -1 and solution == 1:
    #         FN = FN + 1
    #     elif predicted == 1 and solution == -1:
    #         FP = FP + 1
    # print("Confusion Matrix :")
    # confusion_matrix = [[TN, FN], [FP, TP]]
    # for i in confusion_matrix:
    #     print (i)

    accuracy = count / len(tdata)
    return accuracy

#Get updated weights and alpha for the current tree in consideration
def getUpdatedweightAndAlpha(data, tree, weightList):

    #running classifier on test set
    sumError = 0.0
    incorrectIndex = []
    for i in range(len(data)):
        predicted = classify(tree, data[i])
        # print "predicted for",i,"is",predicted
        solution = data[i][0]
        if predicted != solution:
            incorrectIndex.append(i)
            sumError += weightList[i]
    innerTerm = (1-sumError)/sumError
    alpha = 0.5*(math.log(innerTerm))

    #updating weights
    for i in range(len(weightList)):
        if i in incorrectIndex:
            weightList[i] = weightList[i] * math.exp(alpha)
        else:
            weightList[i] = weightList[i] * math.exp(-alpha)

    #normalizing weights
    totalWeight = sum (weightList)
    weightList = [float(x)/totalWeight for x in weightList]

    return alpha, weightList

#Running boosting on decision trees for a specified depth.
def boosting(data, iterations, user_depth):

    weightList = [float(1/(len(data)))] *len(data)

    Boostingtrees = {}

    for i in range(iterations):
        #pass weights with calculate
        newTree = calculate(data, user_depth, 0, weightList)

        alphaDtree, weightList = getUpdatedweightAndAlpha(data, newTree, weightList)
        Boostingtrees[i] = [newTree, alphaDtree]

    return Boostingtrees

def learn_boosted(tdepth, nummodels, datapath):

    train_location = datapath + '/agaricuslepiotatrain1.csv'
    f = open(train_location, 'r')
    data = DataProcessing.preprocess(f)

    test_location = datapath + '/agaricuslepiotatest1.csv'
    t = open(test_location, 'r')

    tdata = DataProcessing.preprocess(t)

    boosting_models = boosting(data, nummodels, tdepth)

    accuracy = classify_accu(boosting_models, tdata)

    print ("Accuracy : ", accuracy)
