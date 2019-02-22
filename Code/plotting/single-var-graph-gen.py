# -*- coding: utf-8 -*-
"""
Title : generating plot of windowed data single variables vs index (time).
Author : Ujjawal.K.Panchal

"""

os.chdir(r'..\..\Plots\Single Attribute Plots')
for feature in range(0,len(feature_names )):
    line_0 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'a']
    line_0_19 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'b']
    line_2_5 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'c']
    line_4_5 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'd']

    plt.plot([i for i in range(len(line_0))], line_0, color = 'brown', label = 'land')

    plt.plot([i for i in range(len(line_0_19))], line_0_19, color = 'magenta', label = '0.19 feet')

    plt.plot([i for i in range(len(line_2_5))], line_2_5, color = 'blue', label = '2.5 feet')

    plt.plot([i for i in range(len(line_4_5))], line_4_5, color = 'red', label = '4.5 feet')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel(feature_names[feature])
    plt.savefig(feature_names[feature]+'vsTime.png')
    plt.show()



