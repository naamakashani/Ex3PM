# This is a sample Python script.
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
classes_num = 9
article_num = 2124

# global parametrs
# estimation step parametrs
# numpy 2-D array of size claaes_num , artical_num
W = np.zeros((article_num, classes_num))

# maximization step parametrs
# probability for 9 classes
alpha = []
# list of dictionaries for 9 classes, for each word what is the Pik
Pik = [{}, {}, {}, {}, {}, {}, {}, {}, {}]

# list of 2124 dictionary for each artical create dictionary of words and their counts
ntk = []

# dictionary of all words in the develop.txt file and their counts (filter out lower than 3)
vocabulary_development = {}


def initial_ntk_and_vocabulary():
    """
    initial ntk and vocabulary_development


    """
    # open develop.txt file and create dictionary for each article of number of words
    with open('develop.txt', 'r') as file:
        lines = file.read().splitlines()
        start_point = lines[2:]
        i = 0
        for artical in start_point:
            if i % 4 == 0:
                word_counts = {}
                words = artical.split()
                for word in words:
                    if word in vocabulary_development:
                        vocabulary_development[word] += 1
                    else:
                        vocabulary_development[word] = 1
                    # If the word is already a key in the dictionary, increment its count
                    if word in word_counts:
                        word_counts[word] += 1
                    # If the word is not in the dictionary, add it with a count of 1
                    else:
                        word_counts[word] = 1
                ntk.append(word_counts)
            i += 1

    # filter out words of vocabulary_development that are less than 3
    for key in list(vocabulary_development.keys()):
        if vocabulary_development[key] < 3:
            del vocabulary_development[key]


def initial_W():
    """
    initial W with modulo-9 manner
    :return:
    """
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
    liklihood_list = []
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
        con_flag = check_stop_criterion(liklihood_list)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    initial_ntk_and_vocabulary()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
