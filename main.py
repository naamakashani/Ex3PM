# Naama Kashani 312400476 Shachar Engelman
import math
import numpy as np
import matplotlib.pyplot as plt

classes_num = 9
article_num = 2124

# define EM parameters
# estimation step parameters
# numpy 2-D array of size classes , article_num
W = np.zeros((article_num, classes_num))

# maximization step parameters
# probability for 9 classes
alpha = [0] * 9
# list of dictionaries for 9 classes, for each word what is the Pik
Pik = [{}, {}, {}, {}, {}, {}, {}, {}, {}]

# list of 2124 dictionary for each article create dictionary of words and their counts
ntk = []

# dictionary of all words in the develop.txt file after filter out lower than 3
vocabulary = set()

# list of given topics
topics = []

# list of header of topics for each article
topics_artical = []
clusters_labels = []


def initial_ntk_and_vocabulary():
    """
    initial ntk and vocabulary_development


    """
    vocabulary_development = {}

    # open develop.txt file and create dictionary for each article of number of words
    with open('develop.txt', 'r') as file:
        lines = file.read().splitlines()
        parts = lines[0].split('\t')[2:]
        topics = []
        for topic in parts:
            # if topic contain '>' remove it
            if '>' in topic:
                topic = topic.split('>')[0]
            topics.append(topic)
        topics_artical.append(topics)
        start_point = lines[2:]
        j = 2
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

            if j % 4 == 0:
                parts = artical.split('\t')[2:]
                topics = []
                for topic in parts:
                    # if topic contain '>' remove it
                    if '>' in topic:
                        topic = topic.split('>')[0]
                    topics.append(topic)
                topics_artical.append(topics)

            i += 1
            j += 1

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
    k = 10
    log_likelihood = 0
    for i in range(article_num):
        Zj = np.zeros(classes_num)
        for j in range(classes_num):
            Zj[j] = np.log(alpha[j]) + np.sum([ntk[i].get(key, 0) * np.log(value) for key, value in Pik[j].items()])

        max_Zj = np.max(Zj)
        Zj -= max_Zj  # Subtract max_Zj from all Zj

        mask = Zj < -k
        W[i] = np.where(mask, 0, np.exp(Zj))
        sum_all_grader = np.sum(W[i])

        W[i] /= sum_all_grader
        log_likelihood += np.log(sum_all_grader) + max_Zj
    return log_likelihood


def alpha_calc():
    epsilon = 0.01
    sum_class = np.sum(W, axis=0, dtype=np.float64)
    alpha[:] = np.where(sum_class == 0, epsilon, sum_class / float(article_num))
    alpha[:] /= (np.sum(alpha))


def p_cal():
    lamda = 0.01
    denominators = np.zeros(classes_num)

    for i in range(classes_num):
        for k in range(article_num):
            denominators[i] += W[k][i] * sum(ntk[k].values())

    for i in range(classes_num):
        for word in vocabulary:
            numerator = 0
            for k in range(article_num):
                if word in ntk[k]:
                    numerator += W[k][i] * ntk[k][word]
            Pik[i][word] = (numerator + lamda) / float(denominators[i] + lamda * len(vocabulary))


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
    # return false if the difference between the last two log_liklihood is less than 10
    if len(log_liklihood_list) > 1:
        if abs(log_liklihood_list[-1] - log_liklihood_list[-2]) < 10:
            return False
    return True


def EM():
    # random W
    maximiztion_step()
    con_flag = True
    liklihood_list = []
    pep_list = []
    while (con_flag):
        log_likelihood= estimation_step()
        maximiztion_step()
        # calc_liklihood
        # log_likelihood = calc_log_liklihood()
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
    plt.savefig('Log_Likelihood.png')
    plt.show()


def plot_perlexity(pep_list):
    plt.plot(pep_list)
    plt.xlabel('Iteration')
    plt.ylabel('Perlexity')
    plt.title('Perlexity')
    plt.savefig('Perlexity.png')
    plt.show()


def hard_assignment():
    """
    for each article take the cluster
    :return:
    """

    hard_assignment_list = np.argmax(W, axis=1)
    return hard_assignment_list


def save_topics():
    """
    save the topics in a file
    :return:
    """
    with open('topics.txt', 'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            topics.append(line)


def get_last_element(line):
    """
    use the function to sort the confustion matrix based on descending order.
    """
    elements = line.strip().split(',')
    return elements[-1]


def create_matrix(hard_assignment_list):
    """
    create the confustion matrix of the articles
    :return:
    """
    matrix = [{topic: 0 for topic in topics} for i in range(classes_num)]
    counter = np.zeros(classes_num)
    for i in range(article_num):
        # find the cluster for the article
        cluster = hard_assignment_list[i]
        counter[cluster] += 1
        # add 1 for all topics in the header of this article
        for topic in topics_artical[i]:
            # add 1 for each topic on the header of the article
            matrix[cluster][topic] += 1

    # assigne topic for each cluster
    for dictionary in matrix:
        max_key = max(dictionary, key=lambda k: dictionary[k])
        clusters_labels.append(max_key)
    # create matrix
    lines = []
    # Iterate over each row in the matrix
    for i, row in enumerate(matrix):
        # Extract counts from the row and convert to string
        counts = ",".join(str(c) for t, c in row.items())
        # Combine index, counts, and total count
        line = f"{i},{counts},{counter[i]}"
        lines.append(line)

    # Create header line with topics
    header_line = "," + ",".join(topics)
    header_line = header_line[1:]
    # sort matrix based on the last element
    sorted_lines = sorted(lines, key=get_last_element, reverse=True)
    # Insert header line at the beginning
    sorted_lines.insert(0, header_line)
    fh = open("matrix.csv", "w")
    fh.write("\n".join(sorted_lines))
    fh.close()


def accuracy_calc(hard_assignment_list):
    """
    calculate accuracy based on hard assingment if it is one of the given topics for article.
    :param hard_assignment_list:
    :return:
    """
    correct = 0
    for artical in range(article_num):
        topic = clusters_labels[hard_assignment_list[artical]]
        # check if the topic is in the topics of the article
        if topic in topics_artical[artical]:
            correct += 1
    accuracy = correct / article_num
    print(accuracy)


def main():
    initial_ntk_and_vocabulary()
    initial_W()
    save_topics()
    liklihood_list, perplexity_list = EM()
    plot_liklihood(liklihood_list)
    plot_perlexity(perplexity_list)
    hard_assignment_list = hard_assignment()
    create_matrix(hard_assignment_list)
    accuracy_calc(hard_assignment_list)


if __name__ == '__main__':
    main()
