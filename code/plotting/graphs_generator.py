###################################
    #purpose - to generate plots
    #creator - Hardik Ajmani
    #date - 22/2/19
################################


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os                       #for path join


color = {0 : 'red', 0.19 : 'blue', 2.5 : 'green', 4.5 : 'orange'}
def slice_by_class(dataset, depth_classes):
    __classes = set(depth_classes)
    sliced_data = {}

    for c in __classes:

        #locating the indices of each class
        loc = np.argwhere(c == depth_classes)

        sliced_data[c] = dataset.iloc[loc[0,0] : loc[-1,0], :].values

    return (sliced_data)


def plot_graph(sliced_data, depth_classes, column_names):

    for i in range(len(column_names) - 1):
        for j in range(len(column_names) - 1):
            if i == j: continue
            for c in depth_classes:
                #if c != 0.19:
                    #plt.plot(range(100), sliced_data[c][:100,j], color = color[c], label = str(c))
                    plt.plot(sliced_data[c][:100,i], sliced_data[c][:100,j], color = color[c], label = str(c))
            plt.xlabel("First Hundred Points")
            plt.ylabel(column_names[j])
            plt.title(("First Hundred points of " + column_names[j]))
            plt.legend()
            plt.rcParams["figure.figsize"] = [15.5, 8.0]
            path = os.path.join("plots", "one feature plots (first 100)", column_names[j])

            plt.savefig((path + '.png'), bbox_inches='tight')
            #plt.show()












df = pd.read_csv(os.path.join("data", "windowed", "window_50_stride_25.csv"))
# it only executes when script run from the main folder

column_names = list(df)
#print(column_names)

depth_classes = df.iloc[:, -1].values


sliced_data = slice_by_class(df, depth_classes)

#seperete data class wise
#plot data in loop with each class name and header

plot_graph(sliced_data, set(depth_classes), column_names)

