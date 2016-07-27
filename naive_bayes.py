# Tyler Sorg
# Machine Learning
# Naive Bayes Classifier using Spambase
import numpy as np


def load_spam_data(filename="spambase.data"):
    """
    Each line in the datafile is a csv with features values, followed by a
    single label (0 or 1), per sample; one
    sample per line
    """

    unprocessed_data_file = file(filename, 'r')

    # Obtain all lines in the file as a list of strings.

    unprocessed_data = unprocessed_data_file.readlines()

    labels = []
    features = []

    for line in unprocessed_data:
        feature_vector = []

        split_line = line.split(',')

        # Iterate across elements in the split_line except
        # for the final element
        for element in split_line[:-1]:
            feature_vector.append(float(element))

        # Add the new vector of feature values for the sample to the features
        # list
        features.append(feature_vector)

        # Obtain the label for the sample and add it to the labels list
        labels.append(int(split_line[-1]))

    "Return List of features with its list of corresponding labels"
    return features, labels


def split_data_training_test(features, labels):
    """
    Divide the data into training and testing sets with close to equal sizes
    and representation of each class.
    """

    spam_count = 0
    not_spam_count = 0

    training_features = []
    training_labels = []
    test_features = []
    test_labels = []

    for feature, label in zip(features, labels):
        if label == 0:
            if not_spam_count % 2 == 0:
                training_features.append(feature)
                training_labels.append(label)
            elif not_spam_count % 2 == 1:
                test_features.append(feature)
                test_labels.append(label)
            not_spam_count += 1

        elif label == 1:
            if spam_count % 2 == 0:
                training_features.append(feature)
                training_labels.append(label)
            elif spam_count % 2 == 1:
                test_features.append(feature)
                test_labels.append(label)
            spam_count += 1
    return training_features, training_labels, test_features, test_labels


def convert_data_to_arrays(features, labels):
    return np.asarray(features), np.asarray(labels)


def classify_test(prior_prob_pos, prior_prob_neg, pos_prod_log_sum,
                  neg_prod_log_sum):
    """
    Given prior probabilities and the products of all the P(
    feature_i|class_j) values, calculate the argmax of the logs
    Return 1 if spam, 0 if not spam
    """
    spam_product_log = np.log(prior_prob_pos) + pos_prod_log_sum
    not_spam_product_log = np.log(prior_prob_neg) + neg_prod_log_sum

    if spam_product_log > not_spam_product_log:
        return 1  # Spam
    elif spam_product_log < not_spam_product_log:
        return 0  # Not spam
    else:
        return np.random.randint(0, 2)


def log_sum_of_probabilities(pos, neg):
    final_pos_product = 0
    final_neg_product = 0
    for pos_prod, neg_prod in zip(pos, neg):
        # To avoid divide by zero warnings in numpy.log
        if pos_prod < 10**-200:
            pos_prod = 10**-1
        if neg_prod < 10**-200:
            neg_prod = 10**-1
        final_pos_product += np.log(pos_prod)
        final_neg_product += np.log(neg_prod)
    return final_pos_product, final_neg_product


def P(test, pos_means, pos_stds, neg_means, neg_stds):
    """
    Return two lists of all the P(feature_i|class_j) values
    """
    pos = []
    for i in range(len(test)):
        pos.append(N(test[i], pos_means[i], pos_stds[i]))
    neg = []
    for i in range(len(test)):
        neg.append(N(test[i], neg_means[i], neg_stds[i]))

    return pos, neg


def N(feature, mean, stddev):
    # if stddev < 0.01:
    #     stddev += 0.01
    coefficient = float(1) / np.sqrt(2 * np.pi * stddev)
    mean_difference = feature - mean
    power = - float((mean_difference ** 2)) / (2 * (stddev ** 2))
    return coefficient * np.exp(power)


def main():
    # PREPROCESSING DATA

    # GET ALL THE DATA
    features, labels = load_spam_data()

    # Compute the prior probabilities
    prior_prob_spam = float(labels.count(1)) / len(labels)
    prior_prob_not_spam = float(labels.count(0)) / len(labels)
    print 'Prior probabilities: Spam=%f, Not spam=%f' % (prior_prob_spam,
                                                         prior_prob_not_spam)

    # SPLIT IN HALF FOR TRAINING AND TESTING
    training_features, training_labels, testing_features, testing_labels = \
        split_data_training_test(features, labels)

    # Separate the positive and negative training features
    training_spam_features = []
    training_not_spam_features = []
    for feature, label in zip(training_features, training_labels):
        if label == 0:
            training_not_spam_features.append(feature)
        elif label == 1:
            training_spam_features.append(feature)

    # CONVERT TO NUMPY ARRAYS
    training_spam_features = np.array(training_spam_features)
    training_not_spam_features = np.array(training_not_spam_features)
    testing_features, testing_labels = convert_data_to_arrays(testing_features,
                                                              testing_labels)

    # Calculate means and standard deviations for each feature in the
    # training set
    spam_means = np.mean(training_spam_features, axis=0)
    spam_stds = np.std(training_spam_features, axis=0)
    not_spam_means = np.mean(training_not_spam_features, axis=0)
    not_spam_stds = np.std(training_not_spam_features, axis=0)

    # Use the Naive Bayes algorithm to classify the instances in your test set
    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(len(testing_labels)):
        pos_probabilities, neg_probabilities = P(testing_features[i],
                                                 spam_means,
                                                 spam_stds, not_spam_means,
                                                 not_spam_stds)
        pos_prod, neg_prod = log_sum_of_probabilities(pos_probabilities,
                                                      neg_probabilities)
        decision = classify_test(prior_prob_spam,
                                 prior_prob_not_spam, pos_prod,
                                 neg_prod)
        label = testing_labels[i]
        if label == 0:
            if decision == 0:
                tn += 1
            elif decision == 1:
                fp += 1
        elif label == 1:
            if decision == 0:
                fn += 1
            elif decision == 1:
                tp += 1

    # confusion_matrix = [tp, fn, fp, tn]
    accuracy = float(tp + tn) / (tp + fn + fp + tn)
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    print 'Confusion matrix:\n\t%d\t%d\n\t%d\t%d' % (tp, fn, fp, tn)
    print 'Accuracy: %f, Precision: %f, Recall: %f' % (accuracy, precision,
                                                       recall)


main()
