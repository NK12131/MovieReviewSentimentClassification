'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py  <corpus directory path> <limit number>

  This version uses cross-validation with the Naive Bayes classifier in NLTK.
  It computes the evaluation measures of precision, recall and F1 measure for each fold.
  It also averages across folds and across labels.
'''
# Importing necessary libraries
import os
import sys
import random
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import sentiment_read_subjectivity # Custom function to read subjectivity lexicons
import sentiment_read_LIWC_pos_neg_words # Custom function to read LIWC word lists

# Set the directory to the Kaggle data directory
dir = '/Users/nithinkumar/Desktop/FinalProjectData'
os.chdir(dir)
## this code is commented off now, but can be used for sentiment lists

# Initialize the positive, neutral, and negative word lists from the subjectivity lexicon
(positivelist, neutrallist, negativelist)     = sentiment_read_subjectivity.read_subjectivity_three_types('./SentimentLexicons/subjclueslen1-HLTEMNLP05.tff')

# Initialize positive and negative word prefix lists from LIWC 
# There is another function 'isPresent' to test if a word's prefix is in the list
(poslist, neglist) = sentiment_read_LIWC_pos_neg_words.read_words()

# Define the feature extraction function
# This function defines features (keywords) of a document for a bag-of-words (BOW) or unigram baseline.
# Each feature is 'V_(keyword)' and is true or false depending on whether that keyword is in the document.
# Define the feature extraction function
def document_features(document, word_features):
    document_words = set(document)  # Convert the document into a set of tokens
    features = {}
    for word in word_features:  # Check if the keyword is in the document
        features['V_{}'.format(word)] = (word in document_words)
    return features

# Cross-validation function 
# This function takes the number of folds, the feature sets, and the labels.
# It iterates over the folds, using different sections for training and testing in turn.
# It prints the performance for each fold and the average performance at the end.
# Cross-validation function
def cross_validation_PRF(num_folds, featuresets, labels):
    num_labels = len(labels)
    subset_size = int(len(featuresets) / num_folds)
    total_precision_list = [0] * len(labels)
    total_recall_list = [0] * len(labels)
    total_F1_list = [0] * len(labels)
    accuracy_list = []

    for i in range(num_folds):
        test_this_round = featuresets[i * subset_size:][:subset_size]
        train_this_round = featuresets[:i * subset_size] + featuresets[(i + 1) * subset_size:]
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        goldlist = [label for (features, label) in test_this_round]
        predictedlist = [classifier.classify(features) for (features, label) in test_this_round]
        
        precision_list, recall_list, F1_list = eval_measures(goldlist, predictedlist, labels)
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        accuracy_list.append(accuracy_this_round)

        for idx in range(len(labels)):
            total_precision_list[idx] += precision_list[idx]
            total_recall_list[idx] += recall_list[idx]
            total_F1_list[idx] += F1_list[idx]

    print('\nAverage Accuracy:', sum(accuracy_list) / num_folds)
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
    for idx, lab in enumerate(labels):
        avg_precision = total_precision_list[idx] / num_folds
        avg_recall = total_recall_list[idx] / num_folds
        avg_F1 = total_F1_list[idx] / num_folds
        print(f"{lab}\t{avg_precision:.3f}\t{avg_recall:.3f}\t{avg_F1:.3f}")

    # Calculate precision, recall, and F1 averaged over all rounds for all labels
    precision_list = [tot/num_folds for tot in total_precision_list]
    recall_list = [tot/num_folds for tot in total_recall_list]
    F1_list = [tot/num_folds for tot in total_F1_list]

    print('\nAverage Accuracy : ', sum(accuracy_list)/num_folds)
    # the evaluation measures in a table with one row per label
    
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
    # Macro average over all labels - treats each label equally
    
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
    
    # Macro average over all labels - treats each label equally
    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', "{:10.3f}".format(sum(precision_list)/num_labels), \
          "{:10.3f}".format(sum(recall_list)/num_labels), \
          "{:10.3f}".format(sum(F1_list)/num_labels))

    # Micro average calculation, weighted by the number of items per label
    label_counts = defaultdict(int)  # Default value for any key is int(), which is 0
    
    # Now, you can safely increment any label count without prior initialization
    for (doc, lab) in featuresets:
     label_counts[lab] += 1
    # make weights compared to the number of documents in featuresets
    num_docs = len(featuresets)
    label_weights = [(label_counts[lab] / num_docs) for lab in labels]
    print('\nLabel Counts', label_counts)
    #print('Label weights', label_weights)
    # print macro average over all labels
    print('Micro Average Precision\tRecall\t\tF1 \tOver All Labels')
    precision = sum([a * b for a,b in zip(precision_list, label_weights)])
    recall = sum([a * b for a,b in zip(recall_list, label_weights)])
    F1 = sum([a * b for a,b in zip(F1_list, label_weights)])
    print( '\t', "{:10.3f}".format(precision), \
      "{:10.3f}".format(recall), "{:10.3f}".format(F1))
    

# Function to compute precision, recall, and F1 for each label
# and for any number of labels
# Input: list of gold labels, list of predicted labels (in the same order)
# Output: returns lists of precision, recall, and F1 for each label
# Function to compute precision, recall, and F1 for each label
def eval_measures(gold, predicted, labels):
    recall_list = []
    precision_list = []
    F1_list = []

    for lab in labels:
        TP = FP = FN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:
                TP += 1
            if val == lab and predicted[i] != lab:
                FN += 1
            if val != lab and predicted[i] == lab:
                FP += 1
        if TP == 0:
            precision = recall = F1 = 0
        else:
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        F1_list.append(F1)
    return precision_list, recall_list, F1_list

# Function to read Kaggle training file, train and test a classifier 
# Function to process the Kaggle training file and train/test classifiers
def processkaggle(dirPath, limitStr):
    limit = int(limitStr)
    f = open(os.path.join(dirPath, 'corpus/train.tsv'), 'r')
    phrasedata = []
    for line in f:
        if not line.startswith('Phrase'):
            line = line.strip()
            phrasedata.append(line.split('\t')[2:4])
    random.shuffle(phrasedata)
    phraselist = phrasedata[:limit]
    
    print(f'Read {len(phrasedata)} phrases, using {len(phraselist)} random phrases')

    phrasedocs = [(nltk.word_tokenize(phrase[0]), int(phrase[1])) for phrase in phraselist]
    all_words_list = [word for (sent, cat) in phrasedocs for word in sent]
    all_words = nltk.FreqDist(all_words_list)
    word_features = [word for (word, count) in all_words.most_common(1500)]
    featuresets = [(document_features(d, word_features), c) for (d, c) in phrasedocs]

    label_list = [c for (d, c) in phrasedocs]
    labels = list(set(label_list))
    num_folds = 5
    cross_validation_PRF(num_folds, featuresets, labels)


"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])
    
    