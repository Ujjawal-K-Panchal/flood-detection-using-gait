from SVM_Classifier_Script import sorted_for_each_plane, X, Y, feature_names
import matplotlib.pyplot as plt
import sys
import random
import numpy as np
import os

feature_indices = []
top_features = []
for plane in sorted_for_each_plane.keys():
    top_features = sorted_for_each_plane[plane][:2]
    feature_indices = [feature_names.index(f[0]) for f in top_features]

    #example:- top_features = [['LINEAR_ACCELERATION_Y_median', 0.3054192514922668], ['GRAVITY_Y_mean', 0.27736765414504283]]
    class_0 =    [[X[i,feature_indices[0]],X[i,feature_indices[1]]] for i in range(0,len(X)) if Y[i] == 'a']
    class_0_19 = [[X[i,feature_indices[0]],X[i,feature_indices[1]]] for i in range(0,len(X)) if Y[i] == 'b']
    class_2_5 =  [[X[i,feature_indices[0]],X[i,feature_indices[1]]] for i in range(0,len(X)) if Y[i] == 'c']
    class_4_5 =  [[X[i,feature_indices[0]],X[i,feature_indices[1]]] for i in range(0,len(X)) if Y[i] == 'd']

    random_indices = random.sample(range(139), 100)
    #print(random_indices)
    class_0_feature_0 =    [class_0[i][0] for i in random_indices]
    class_0_feature_1 =    [class_0[i][1] for i in random_indices]
    class_0_19_feature_0 = [class_0_19[i][0] for i in random_indices]
    class_0_19_feature_1 = [class_0_19[i][1] for i in random_indices]
    class_2_5_feature_0 =  [class_2_5[i][0] for i in random_indices]
    class_2_5_feature_1 =  [class_2_5[i][1] for i in random_indices]
    class_4_5_feature_0 =  [class_4_5[i][0] for i in random_indices]
    class_4_5_feature_1 =  [class_4_5[i][1] for i in random_indices]
    
    
    plt.scatter(class_0_feature_0, class_0_feature_1, color = 'yellow', label = 'land')
    plt.scatter(class_0_19_feature_0, class_0_19_feature_1, color = 'magenta', label = '0.19 feet')
    plt.scatter(class_2_5_feature_0, class_2_5_feature_1, color = 'blue', label = '2.5 feet')
    plt.scatter(class_4_5_feature_0, class_4_5_feature_1, color = 'red', label = '4.5 feet')
    plt.legend()

    title = "Plane_" + str(plane) + "____" + top_features[0][0]+"_vs_" + top_features[1][0]
    path = os.path.join("plots", "two", title)
    plt.suptitle(title)
    plt.xlabel(top_features[0][0])
    plt.ylabel(top_features[1][0])
    plt.savefig(path + ".png")
    #plt.show()
    