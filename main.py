# Naama Kashani 312400476 Shachar Engelman
import math
import numpy as np

classes_num = 9
article_num = 2124

# define EM parameters
# estimation step parameters
# numpy 2-D array of size classes , article_num
W = np.zeros((article_num, classes_num))

# maximization step parameters
# probability for 9 classes
alpha = [0]*9
# list of dictionaries for 9 classes, for each word what is the Pik
Pik = [{}, {}, {}, {}, {}, {}, {}, {}, {}]

# list of 2124 dictionary for each article create dictionary of words and their counts
ntk = []

# dictionary of all words in the develop.txt file after filter out lower than 3
vocabulary = set()


def initial_ntk_and_vocabulary():
    """
    initial ntk and vocabulary_development


    """
    vocabulary_development = {}
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
        if vocabulary_development[key] <= 3:
            del vocabulary_development[key]
    vocabulary.update(vocabulary_development.keys())


def initial_W():
    """
    initial W with modulo-9 manner
    :return:
    """
    for i in range(article_num):
        W[i, i % classes_num] = 1


def estimation_step():
    # update the W values using Underflow Scaling
    k = 10
    # for each artical update the Wi
    for i in range(article_num):
        list_Zj = []
        # run for all classes
        for j in range(classes_num):
            # numerator of the Wi
            Zit = math.log((alpha[j]))
            for key, value in Pik[j].items():
                if key in ntk[i]:
                    ntk_value = ntk[i][key]
                else:
                    ntk_value = 0
                Zit += float(ntk_value * math.log(value))
            list_Zj.append(Zit)

        max_Zit = max(list_Zj)
        # subtract max_Zit from all Zit
        new_list_Zj = [x - max_Zit for x in list_Zj]
        sum_all_grader = 0
        for m in range(classes_num):

            if new_list_Zj[m] < -1 * k:
                W[i][m] = 0
            else:
                W[i][m] = math.exp(new_list_Zj[m])
                sum_all_grader += W[i][m]
        for m in range(classes_num):
            W[i][m] = float(W[i][m]) / sum_all_grader


def alpha_calc():
    epsilon = 0.01
    for i in range(classes_num):
        sum_class = 0
        for j in range(article_num):
            sum_class += W[j][i]
        if sum_class == 0:
            alpha[i] = epsilon
        else:
            alpha[i] = sum_class / article_num
    # normalize alpha list
    sum_alpha = sum(alpha)
    for i in range(classes_num):
        alpha[i] = float(alpha[i]) / sum_alpha


def p_cal():
    lamda = 0.01
    for i in range(classes_num):
        for word in vocabulary:
            numerator = 0
            denominator = 0
            for k in range(article_num):
                if word in ntk[k]:
                    numerator += W[k][i] * (ntk[k][word])
                denominator += W[k][i] * sum(ntk[k].values())
            Pik[i][word] = float(numerator + lamda) / (denominator + lamda * len(vocabulary))


def maximiztion_step():
    alpha_calc()
    p_cal()


def calc_log_liklihood():
    ln_L = 0
    k = 10
    # for each artical update the Wi
    for i in range(article_num):
        list_Zj = []
        # run for all classes
        for j in range(classes_num):
            # numerator of the Wi
            Zit = math.log((alpha[j]))
            for key, value in Pik[j].items():
                if key in ntk[i]:
                    ntk_value = ntk[i][key]
                else:
                    ntk_value = 0
                Zit += float(ntk_value * math.log(value))
            list_Zj.append(Zit)

        max_Zit = max(list_Zj)
        # subtract max_Zit from all Zit
        new_list_Zj = [x - max_Zit for x in list_Zj]
        sum_all_grader = 0
        for m in range(classes_num):
            if new_list_Zj[m] >= -1 * k:
                sum_all_grader += math.exp(new_list_Zj[m])
        ln_L += max_Zit + math.log(sum_all_grader)
    return ln_L


def clac_perlexity(log_likelihood):
    # perlexity = e ^ (-1 /N * log_likelihood). N ia the number of words
    preplexity = math.exp(-1 / len(vocabulary) * log_likelihood)
    return preplexity



def check_stop_criterion(log_liklihood_list):
    # return false if the difference between the last two log_liklihood is less than 0.1
    if len(log_liklihood_list) > 1:
        if abs(log_liklihood_list[-1] - log_liklihood_list[-2]) < 0.1:
            return False
    return True




def EM():
    # random W
    initial_W()
    maximiztion_step()
    con_flag = True
    liklihood_list = []
    pep_list = []
    while (con_flag):
        estimation_step()
        maximiztion_step()

        # calc_liklihood
        log_likelihood = calc_log_liklihood()
        liklihood_list.append(log_likelihood)

        print("log_likelihood", log_likelihood)
        # calc_perlexity
        perlexity = clac_perlexity(log_likelihood)
        pep_list.append(perlexity)
        print("perlexity", perlexity)
        con_flag = check_stop_criterion(liklihood_list)
    return liklihood_list, pep_list

def plot_liklihood(liklihood_list):
    plt.plot(liklihood_list)
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood')
    plt.show()
    #save the plot
    plt.savefig('Log_Likelihood.png')

def plot_perlexity(pep_list):
    plt.plot(pep_list)
    plt.xlabel('Iteration')
    plt.ylabel('Perlexity')
    plt.title('Perlexity')
    plt.show()
    #save the plot
    plt.savefig('Perlexity.png')

def hard_assignment():
    #run on W and for each artical take the max value
    hard_assignment_list = []
    for i in range(article_num):
        max_index = W[i].index(max(W[i]))
        hard_assignment_list.append(max_index)
    return hard_assignment_list




def main():
    initial_ntk_and_vocabulary()
    initial_W()
    liklihood_list, perplexity_list = EM()
    plot_liklihood(liklihood_list)
    plot_perlexity(perplexity_list)
    hard_assignment_list = hard_assignment()



if __name__ == '__main__':
    main()
