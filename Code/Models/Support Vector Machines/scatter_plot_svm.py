from SVM_Classifier_Script import sorted_for_each_plane, X, Y, feature_names
import matplotlib.pyplot as plt
import sys
import random
import numpy as np
import os
from pylab import rcParams
#rcParams['figure.figsize'] = 3.1,2.2

feature_indices = []
top_features = []
plane_wise_class = { 0 : ['a','b'], 1 : ['a', 'c'], 2: ['a','d'], 3 : ['b', 'c'], 4 : ['b', 'd'], 5 : ['c', 'd'] }
class_depth = {'a' : 'land', 'b' : '0.19 feet', 'c' : '2.5 feet', 'd' : '4.5 feet' }
class_wise_color = {'a' : 'blue', 'b' : 'green', 'c' : 'red', 'd': 'yellow'}
for plane in sorted_for_each_plane.keys():
    top_features = sorted_for_each_plane[plane][:2]
    feature_indices = [feature_names.index(f[0]) for f in top_features]

    #example:- top_features = [['LINEAR_ACCELERATION_Y_median', 0.3054192514922668], ['GRAVITY_Y_mean', 0.27736765414504283]]
    class_1 = [[X[i,feature_indices[0]],X[i,feature_indices[1]]] for i in range(0,len(X)) if Y[i] == plane_wise_class[plane][0]]
    class_2 = [[X[i,feature_indices[0]],X[i,feature_indices[1]]] for i in range(0,len(X)) if Y[i] == plane_wise_class[plane][1]]

    random_indices = random.sample(range(139), 100)
    #print(random_indices)
    class_1_feature_0 = [class_1[i][0] for i in random_indices]
    class_1_feature_1 = [class_1[i][1] for i in random_indices]
    class_2_feature_0 = [class_2[i][0] for i in random_indices]
    class_2_feature_1 = [class_2[i][1] for i in random_indices]

    
    
    plt.scatter(class_1_feature_0, class_1_feature_1,s = 7, color = class_wise_color[plane_wise_class[plane][0]], label = class_depth[plane_wise_class[plane][0]])
    plt.scatter(class_2_feature_0, class_2_feature_1,s = 7, color = class_wise_color[plane_wise_class[plane][1]], label = class_depth[plane_wise_class[plane][1]])
    plt.legend()

    title = top_features[0][0]+"_vs_" + top_features[1][0]
    path = os.path.join("plots", "two", "plot_" + str(plane))
    plt.suptitle(title)
    plt.xlabel(top_features[0][0])
    plt.ylabel(top_features[1][0])
    #plt.savefig(path + ".png")
    plt.show()
    