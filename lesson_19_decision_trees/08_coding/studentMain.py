#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify
import graphviz
from sklearn import tree

from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)




prediction_test = clf.predict(features_test)

acc = accuracy_score(labels_test, prediction_test)

print('accuracy:', acc)


#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)



dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

output_image("test.png", "png", open("test.png", "rb").read())
