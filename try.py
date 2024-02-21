import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
classes_num = 9
article_num = 2124

# global parametrs
# numpy 2-D array of size claaes_num , artical_num
W = np.zeros((article_num, classes_num))

# maximization step parametrs
# probability for 9 classes
alpha = []
# list of dictionaries for 9 classes, for each word what is the Pik
Pik = [{}, {}, {}, {}, {}, {}, {}, {}, {}]

if __name__ == '__main__':

    for i in range(article_num):
        W[i,i % classes_num] = 1

    # print the W matrix
    print(W)
