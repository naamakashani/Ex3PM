# This is a sample Python script.
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
classes_num = 9
article_num = 2124

# global parametrs
#estimation step parametrs
# numpy 2-D array of size claaes_num , artical_num
W = np.zeros((article_num, classes_num))

# maximization step parametrs
# probability for 9 classes
alpha = []
# list of dictionaries for 9 classes, for each word what is the Pik
Pik = [{}, {}, {}, {}, {}, {}, {}, {}, {}]

#list of 2124 dictionary
ntk=[]
def initial_ntk():
    pass







def initial_W():
    for i in range(article_num):
        W[i, i % classes_num] = 1

def estimation_step():
    pass
def maximiztion_step():
    pass
def calc_liklihood():
    pass

def clac_perlexity():
    pass
def check_stop_criterion():
    pass


def EM():
    # random W
    initial_W()
    maximiztion_step()
    con_flag = True
    liklihood_list=[]
    while (con_flag):
        estimation_step()
        maximiztion_step()

        # calc_liklihood
        likelihood = calc_liklihood()
        liklihood_list.append(likelihood)

        print(likelihood)
        # calc_perlexity
        perlexity = clac_perlexity()
        print(perlexity)
        con_flag= check_stop_criterion(liklihood_list)



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
