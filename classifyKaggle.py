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
'''
# open python and nltk packages needed for processing
import os
import sys
import random
from xml.sax.handler import feature_external_ges
import nltk
import re
from nltk.corpus import stopwords
import sentiment_read_subjectivity
import sentiment_read_LIWC_pos_neg_words
import classifyKaggle_crossval
from nltk.metrics import ConfusionMatrix
from nltk.collocations import *
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from nltk.util import ngrams

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# Comprehensive stopwords list
def get_extended_stopwords():
    nltkstopwords = set(stopwords.words('english'))
    custom_stopwords = {'could', 'would', 'might', 'must', 'need', 'sha', 'wo', 'y', "'s", "'d", 
                        "'ll", "'t", "'m", "'re", "'ve", "n't", "'i", 'not', 'no', 'can', 'don', 
                        'nt', 'actually', 'also', 'always', 'even', 'ever', 'just', 'really', 
                        'still', 'yet', 'however', 'nevertheless', 'furthermore', 'therefore', 
                        'otherwise', 'meanwhile', 'though', 'although', 'thus', 'hence', 'indeed', 
                        'perhaps', 'especially', 'specifically', 'usually', 'often', 'sometimes', 
                        'certainly', 'typically', 'mostly', 'generally', 'about', 'above', 'across', 
                        'after', 'against', 'among', 'around', 'at', 'before', 'behind', 'below', 
                        'beneath', 'beside', 'between', 'beyond', 'during', 'inside', 'onto', 
                        'outside', 'through', 'under', 'upon', 'within', 'without'}
    return nltkstopwords.union(custom_stopwords)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stopwords_set = get_extended_stopwords()

# Initialize lists for positive, neutral, and negative words using a custom module for reading subjectivity classifications
(positivelist, neutrallist, negativelist) = sentiment_read_subjectivity.read_subjectivity_three_types('/Users/nithinkumar/Desktop/Sentiment-Classification-of-Movie-Reviews-master/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff')


# Initialize lists for positive and negative word prefixes from LIWC (Linguistic Inquiry and Word Count), an important tool for psychological and linguistic analysis
# A function isPresent is also defined in the module to check if a word's prefix matches any in the lists
(poslist, neglist) = sentiment_read_LIWC_pos_neg_words.read_words()

# Initialize the sentiment lexicons
dpath = '/Users/nithinkumar/Desktop/FinalProjectData/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff'
SL = sentiment_read_subjectivity.readSubjectivity(dpath)

# Enhance preprocessing with bi-grams and sentiment scores
def preprocessing(line):
    # Tokenize the input line after converting it to lowercase.
    tokens = word_tokenize(line.lower())
    cleaned_tokens = []
    # Process each token to remove punctuation, check against stopwords, and lemmatize.
    for token in tokens:
          # Remove punctuation from the token.
        cleaned_token = re.sub(r'[^\w\s]', '', token)
        
        # Check if the cleaned token is not in the stopwords list.
        if cleaned_token and cleaned_token not in stopwords_set:
            # Lemmatize the token to reduce it to its base form.
            lemmatized_token = lemmatizer.lemmatize(cleaned_token)
            
            # Only include tokens longer than 2 characters.
            if len(lemmatized_token) > 2:
                cleaned_tokens.append(lemmatized_token)
    # Add bi-grams
    # Generate bi-grams from the list of cleaned and lemmatized tokens.
    bi_grams = list(ngrams(cleaned_tokens, 2))
    # Join the words in each bi-gram with an underscore.
    bi_grams = ['_'.join(bigram) for bigram in bi_grams]
    # Combine the single tokens and bi-grams into one list.
    final_tokens = cleaned_tokens + bi_grams
    # Return all tokens as a single string separated by spaces.
    return ' '.join(final_tokens)
  
def filtered_tokens(phrase):
      # Extract tokens and sentiment score from the input tuple.
    tokens, sentiment = phrase # Assuming `phrase` is a tuple of (tokens, sentiment)
    # Filter out tokens that are in the stopwords set.
    filtered = [token for token in tokens if token not in stopwords_set]
    # Return the filtered tokens and the original sentiment as a tuple.
    return (filtered, sentiment)


# Different Functions for feature sets :
#Bag of wordds
def bw(a,i):
  # Calculate the frequency distribution of elements in the list 'a'.
  a = nltk.FreqDist(a)
  # Extract the 'i' most common elements from the frequency distribution.
  # 'w' represents the word and 'c' represents its count in the list.
  wf = [w for (w,c) in a.most_common(i)]
  # Return the list of the most frequent words
  return wf   


#Unigram
def uf(d,wf):
  # Create a set from the list 'd' to allow faster membership testing.
  df= set(d)
  # Initialize an empty dictionary to store unigram features.
  f = {}
  # Loop through each word in the list of words 'wf'
  for word in wf:
    # Create a feature key for each word and assign a boolean value
    # indicating whether the word is present in the data 'd'.
    f['V_%s'% word] = (word in df)
  # Return the dictionary containing feature presence information.
  return f


#Bigram
def bigram_bow(wordlist,n):
  # Utilize NLTK to measure bigram associations.
  bigram_measure = nltk.collocations.BigramAssocMeasures()
  # Create a BigramCollocationFinder from a list of words
  finder = BigramCollocationFinder.from_words(wordlist)
  # Apply a frequency filter to bigrams that appear at least 2 times.
  finder.apply_freq_filter(2)
  # Find the best 4000 bigrams using the chi-squared association measure.
  b_features = finder.nbest(bigram_measure.chi_sq,4000)
  # Return the top 'n' bigrams from the filtered list.
  return b_features[:n]


def bf(doc,word_features,bigram_feature):
  # Create a set of words and a set of bigrams from the document for fast lookup.
  dw = set(doc)
  db = nltk.bigrams(doc)
  # Initialize an empty dictionary to store features
  features = {}
  # Check each word in the word_features list and add it as a binary feature
  for word in word_features:
    features['V_{}'.format(word)] = (word in dw)
  # Check each bigram in the bigram_feature list and add it as a binary feature
  for b in bigram_feature:
    features['B_{}_{}'.format(b[0],b[1])] = (b in db)
  # Return the dictionary containing all word and bigram features.
  return features



#POs Tags
def pf(document, word_features):
    # Create a set of all words in the document for fast lookup.
    document_words = set(document)
    # Generate POS tags for each word in the document.
    tagged_words = nltk.pos_tag(document)
    # Initialize an empty dictionary to store features.
    features = {}
    # Loop through each word in word_features and add it as a binary feature indicating presence in the document.
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    # Initialize counters for different POS tags.
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    # Count occurrences of each major part of speech.
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    # Store counts of each part of speech in the features dictionary.
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    # Return the dictionary containing all the features.
    return features
  



#Sentiment Lexicon Features 
def slf(document, word_features, SL):
    # Create a set of all words in the document for quick lookup.
    document_words = set(document)
    # Create dictionary features with keys like 'V_word' where each key indicates whether a word is present in the document.
    features = {f'V_{word}': (word in document_words) for word in word_features}
    # Initialize a dictionary to count occurrences of sentiment-related terms.
    sentiment_counts = {}
    # Loop through each word in the document.
    for word in document_words:
        # Check if the word is in the sentiment lexicon (SL).
        if word in SL:
            # Extract sentiment attributes for the word.
            strength, posTag, isStemmed, polarity = SL[word]
            # Create a composite key from the strength and polarity (e.g., 'strongPositive').
            key = f'{strength}{polarity.capitalize()}'
            # Initialize the count for this key if it's not already present.
            if key not in sentiment_counts:
                sentiment_counts[key] = 0
            # Increment the count for this sentiment key.
            sentiment_counts[key] += 1

   # Identify keys that indicate positive, negative, and neutral sentiments.
    positive_keys = [key for key in sentiment_counts if 'Positive' in key]
    negative_keys = [key for key in sentiment_counts if 'Negative' in key]
    neutral_keys = [key for key in sentiment_counts if 'Neutral' in key]
    
    # Calculate the total counts for positive, negative, and neutral terms.
    features['positivecount'] = sum(sentiment_counts[key] for key in positive_keys)
    features['negativecount'] = sum(sentiment_counts[key] for key in negative_keys)
    features['neutralcount'] = sum(sentiment_counts[key] for key in neutral_keys if 'Neutral' in key)
    # Return the dictionary containing all the features.
    return features


#Linguistic Inquiry and Word Count
def liwc(doc,word_features,poslist,neglist):
  # Create a set of words from the document to eliminate duplicates and allow for fast lookup.
  doc_words = set(doc)
  # Initialize a dictionary to store features.
  features= {}
  # Loop through each word in word_features and add it as a binary feature indicating presence in the document.
  for word in word_features:
    features['contains({})'.format(word)] = (word in doc_words)
  # Initialize counters for positive and negative words.
  pos = 0
  neg = 0
  # Count occurrences of words listed in the positive and negative lists.
  for word in doc_words:
    if sentiment_read_LIWC_pos_neg_words.isPresent(word,poslist):
      pos+=1
    elif sentiment_read_LIWC_pos_neg_words.isPresent(word,neglist):
      neg+=1
    # Store the counts of positive and negative words in the features dictionary.
    features ['positivecount'] = pos
    features ['negativecount'] = neg

  # Ensure that features for positive and negative counts are present, even if their count is zero.
  if 'positivecount' not in features:
    features['positivecount'] = 0
  if 'negativecount' not in features:
    features['negativecount'] = 0
  # Return the dictionary containing all the features.
  return features




def combo(doc, word_features, SL, poslist, neglist):
    # Create a set of words from the document to eliminate duplicates and allow for fast lookup.
    doc_words = set(doc)
    # Generate a dictionary of binary features indicating the presence of specific words from word_features.
    features = {f'contains({word})': (word in doc_words) for word in word_features}
    # Initialize a dictionary to keep track of different sentiment counts.
    sentiment_counts = {}
    # Count the occurrences of words from poslist and neglist as strong positive or negative sentiments.
    for word in doc_words:
        if word in poslist:
            sentiment_counts['strongPos'] = sentiment_counts.get('strongPos', 0) + 1
        elif word in neglist:
            sentiment_counts['strongNeg'] = sentiment_counts.get('strongNeg', 0) + 1
        # If the word is in the sentiment lexicon, analyze its sentiment attributes.
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            key = f'{strength}{polarity.capitalize()}'
            if key not in sentiment_counts:
                sentiment_counts[key] = 0
            sentiment_counts[key] += 1

    # Gather keys that indicate positive, negative, and neutral sentiments.
    positive_keys = [key for key in sentiment_counts if 'Positive' in key]
    negative_keys = [key for key in sentiment_counts if 'Negative' in key]
    neutral_keys = [key for key in sentiment_counts if 'Neutral' in key]
    
    # Calculate the total counts for positive, negative, and neutral terms.
    features['positivecount'] = sum(sentiment_counts[key] for key in positive_keys)
    features['negativecount'] = sum(sentiment_counts[key] for key in negative_keys)
    features['neutralcount'] = sum(sentiment_counts[key] for key in neutral_keys if 'Neutral' in key)
    # Return the dictionary containing all the features.
    return features





# Saving feature sets for for other classifier training
def save(features, path):
    # Open a file for writing at the specified path.
    f = open(path, 'w')
    # Retrieve the names of the features from the first feature set in the list.
    featurenames = features[0][0].keys()
    # Prepare the header line by cleaning up feature names and appending them into a single string.
    fnameline = ''
    for fname in featurenames:
        # Replace problematic characters in feature names to avoid issues in CSV format.
        fname = fname.replace(',','COM')
        fname = fname.replace("'","SQ")
        fname = fname.replace('"','DQ')
        fnameline += fname + ','
    # Add a label for the class/category column at the end of the header.
    fnameline += 'Level'
    f.write(fnameline)
    f.write('\n')
    # Iterate over each feature set and corresponding class/category.
    for fset in features:
        featureline = ''
        # Gather all feature values into a line, handling missing keys if necessary.
        for key in featurenames:
            # Check if the key exists in the feature set
            if key in fset[0]:
                featureline += str(fset[0][key]) + ','
            else:
                featureline += 'NA,'  # Write 'NA' for missing features in some records.
        # Append the class/category description based on the numerical label.
        if fset[1] == 0:
          featureline += str("Less Negitive")
        elif fset[1] == 1:
          featureline += str("Strong negitive")
        elif fset[1] == 2:
          featureline += str("Neutral")
        elif fset[1] == 3:
          featureline += str("Strongly positive")
        elif fset[1] == 4:
          featureline += str("Less positive")
        # Write the complete line for the current feature set to the file
        f.write(featureline)
        f.write('\n')
    # Close the file after writing all the data.
    f.close()



def naivebayesaccuracy(features):
    split_ratio = 0.1
    cutoff = int(len(features) * split_ratio)
    train_set, test_set = features[cutoff:], features[:cutoff]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    accuracy = nltk.classify.accuracy(classifier, test_set)
    print("\nAccuracy: {:.2%}\n".format(accuracy))

    actual_labels = [label for (features, label) in test_set]
    predicted_labels = [classifier.classify(features) for (features, _) in test_set]

    print(ConfusionMatrix(actual_labels, predicted_labels))





def dt(featuresets):
    cutoff = int(0.1 * len(featuresets))
    train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]
    classifier_dt = SklearnClassifier(DecisionTreeClassifier())
    classifier_dt.train(train_set)

    accuracy = nltk.classify.accuracy(classifier_dt, test_set)
    print("Classifier: Decision Tree\nAccuracy: {:.2%}".format(accuracy))







def svm(featuresets):
    cutoff = int(0.1 * len(featuresets))
    train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]
    classifier_svm = SklearnClassifier(SVC())
    classifier_svm.train(train_set)

    accuracy = nltk.classify.accuracy(classifier_svm, test_set)
    print("Classifier: SVM\nAccuracy: {:.2%}".format(accuracy))
    
    

def rf(featuresets):
    cutoff = int(0.1 * len(featuresets))
    train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]
    classifier_rf = SklearnClassifier(RandomForestClassifier())
    classifier_rf.train(train_set)

    accuracy = nltk.classify.accuracy(classifier_rf, test_set)
    print("Classifier: Random Forest\nAccuracy: {:.2%}".format(accuracy))







def plot_sentiment_distribution(phrasedata):
    sentiments = [int(phrase[1]) for phrase in phrasedata]
    plt.hist(sentiments, bins=5, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Sentiment Label')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentiment Labels')
    plt.xticks(range(0, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_word_frequency_distribution(tokens):
    freq_dist = nltk.FreqDist(tokens)
    plt.figure(figsize=(10, 6))
    freq_dist.plot(30, cumulative=False)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title('Word Frequency Distribution')
    plt.grid(True)
    plt.show()


def plot_top_words(tokens, n=20):
    freq_dist = nltk.FreqDist(tokens)
    top_n = freq_dist.most_common(n)
    words, frequencies = zip(*top_n)
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title(f'Top {n} Words Frequency Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()






def plot_histogram(data):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Word Lengths')
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.show()


def plot_word_length_distribution(tokens):
    word_lengths = [len(word) for word in tokens]
    plt.figure(figsize=(10, 6))
    plt.hist(word_lengths, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.title('Word Length Distribution')
    plt.grid(True)
    plt.show()

def plot_word_length_boxplot(tokens):
    word_lengths = [len(word) for word in tokens]
    plt.figure(figsize=(8, 6))
    plt.boxplot(word_lengths, vert=False)
    plt.xlabel('Word Length')
    plt.title('Box Plot of Word Lengths')
    plt.grid(True)
    plt.show()








# define a feature definition function here

# use NLTK to compute evaluation measures from a reflist of gold labels
#    and a testlist of predicted labels for all labels in a list
# returns lists of precision and recall for each label


# function to read kaggle training file, train and test a classifier 
# function to read kaggle training file, train and test a classifier 
# function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
  f = open('/Users/nithinkumar/Desktop/FinalProjectData/corpus/train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:

    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore th
      # e phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])

  
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')

  for phrase in phraselist[:10]:
    print (phrase)
  
  ## Initialize lists to hold processed and unprocessed data.
  withpreprocessing = []
  withoutpreprocessing= []
  # Iterate through each phrase and sentiment tuple in the phraselist.
  for p in phraselist:
    # Tokenize the first element (phrase) of the tuple using NLTK's word_tokenize
    tokens = nltk.word_tokenize(p[0])
    # Append the tokenized phrase and its associated integer sentiment score to the list without preprocessing.
    withoutpreprocessing.append((tokens, int(p[1])))

    # Apply the preprocessing function to the phrase and re-tokenize the result.
    p[0] = preprocessing(p[0]) # Preprocess the text of the phrase.
    tokens = nltk.word_tokenize(p[0])
    # Append the re-tokenized, preprocessed phrase and its sentiment to the list with preprocessing.
    withpreprocessing.append((tokens, int(p[1])))
  
  # Initialize an empty list to store preprocessed data after further filtering. 
  withpreprocessing_filter=[]
  # Copy each element from the list 'withpreprocessing' to 'withpreprocessing_filter'.
  for p in withpreprocessing:
    withpreprocessing_filter.append(p)
  # Initialize lists to store all tokens collected from processed data.
  filtered_tokens =[]
  unfiltered_tokens = []
  # Iterate over the list that contains preprocessed data.
  for (d,s) in  withpreprocessing_filter:
    # Collect all tokens from the tuple where 'd' is a list of tokens and 's' is the sentiment score.
    for i in d:
      filtered_tokens.append(i)
  # Iterate over the list containing unprocessed data.
  for (d,s) in withoutpreprocessing:
    # Collect all tokens from the tuple where 'd' is a list of tokens and 's' is the sentiment score.
    for i in d:
      unfiltered_tokens.append(i)
  
  
  plot_sentiment_distribution(phrasedata)
  plot_histogram(filtered_tokens)
  plot_top_words(filtered_tokens)
  plot_word_length_distribution(filtered_tokens)
  plot_word_length_boxplot(filtered_tokens)
  
 
  
  


  # continue as usual to get all words and create word features
  
  # feature sets from a feature definition function

  # Apply the 'bw' function to 'filtered_tokens' to obtain the 350 most frequent words.
  filtered_bow_features = bw(filtered_tokens,350)
  # Apply the 'bw' function to 'unfiltered_tokens' to obtain the 350 most frequent words.
  unfiltered_bow_features = bw(unfiltered_tokens,350)

  # Generate unigram features for each tuple in 'withpreprocessing_filter' using the 'filtered_tokens' as the vocabulary.
  # 'd' is the list of tokens, and 's' is the sentiment score from each tuple in 'withpreprocessing_filter'.
  filtered_unigram_features = [(uf(d,filtered_tokens),s) for (d,s) in withpreprocessing_filter]
  # Similarly, generate unigram features for each tuple in 'withoutpreprocessing' using the 'unfiltered_tokens' as the vocabulary.
  # This considers all original tokens without preprocessing to compare the effect.
  unfiltered_unigram_features = [(uf(d,unfiltered_tokens),s) for (d,s) in withoutpreprocessing]

  # Generate bigram features for each tuple in 'withpreprocessing_filter' using filtered tokens and bow features.
  filtered_bigram_features = [(bf(d,filtered_bow_features,bigram_bow(filtered_tokens,350)),s) for (d,s) in withpreprocessing_filter]
  # Similarly, generate bigram features for each tuple in 'withoutpreprocessing' using unfiltered tokens and bow features.
  unfiltered_bigram_features = [(bf(d,unfiltered_bow_features,bigram_bow(unfiltered_tokens,350)),s) for (d,s) in withoutpreprocessing]

  # Generate POS tag features for each tuple in 'withpreprocessing_filter' using the filtered BOW features.
  filtered_pos_features = [(pf(d,filtered_bow_features),s) for (d,s) in withpreprocessing_filter]
  # Similarly, generate POS tag features for each tuple in 'withoutpreprocessing' using the unfiltered BOW features.
  unfiltered_pos_features = [(pf(d,unfiltered_bow_features),s) for (d,s) in withoutpreprocessing]

  # Generate sentiment-related features for each tuple in 'withpreprocessing_filter' using the filtered BOW features and a sentiment lexicon (SL).  
  filtered_sl_features = [(slf(d, filtered_bow_features, SL), c) for (d, c) in withpreprocessing_filter]
  # Similarly, generate sentiment-related features for each tuple in 'withoutpreprocessing' using the unfiltered BOW features and the same sentiment lexicon (SL).
  unfiltered_sl_features = [(slf(d, unfiltered_bow_features, SL), c) for (d, c) in withoutpreprocessing]


  # Apply the liwc function to each document in the preprocessed  data sets.
  filtered_liwc_features = [(liwc(d, filtered_bow_features, poslist,neglist), c) for (d, c) in withpreprocessing_filter]
  # Apply the liwc function to each document in the unprocessed data sets.
  unfiltered_liwc_features = [(liwc(d, unfiltered_bow_features, poslist,neglist), c) for (d, c) in withoutpreprocessing]

  # Apply the combo function to each document in the preprocessed datasets.
  filtered_combo_features =  [(combo(d, filtered_bow_features,SL, poslist,neglist), c) for (d, c) in withpreprocessing_filter]
  # Apply the combo function to each document in the unprocessed datasets.
  unfiltered_combo_features = [(combo(d, unfiltered_bow_features,SL, poslist,neglist), c) for (d, c) in withoutpreprocessing]



  #Saving features
  #savingfeatures(filtered_bow_features,'filtered_bow.csv')
  #savingfeatures(unfiltered_bow_features,'unfiltered_bow.csv')
  
  ## Save unigram features from filtered data to 'filtered_unigram.csv'.
  save(filtered_unigram_features,'filtered_unigram.csv')
  # Save unigram features from unfiltered data to 'unfiltered_unigram.csv'.
  save(unfiltered_unigram_features,'unfiltered_unigram.csv')

  # Save bigram features from filtered data to 'filtered_bigram.csv'.
  save(filtered_bigram_features,'filtered_bigram.csv')
  # Save bigram features from unfiltered data to 'unfiltered_bigram.csv'.
  save(unfiltered_bigram_features,'unfiltered_bigram.csv')

  # Save POS tag-based features from filtered data to 'filtered_pos.csv'.
  save(filtered_pos_features,'filtered_pos.csv')
  #Save POS tag-based features from unfiltered data to 'unfiltered_pos.csv'.
  save(unfiltered_pos_features,'unfiltered_pos.csv')

  # Save sentiment lexicon features from filtered data to 'filtered_sl.csv'.
  save(filtered_sl_features,'filtered_sl.csv')
  # Save sentiment lexicon features from unfiltered data to 'unfiltered_sl.csv'.
  save(unfiltered_sl_features,'unfiltered_sl.csv')

  # Save LIWC-based features from filtered data to 'filtered_liwc.csv'.
  save(filtered_liwc_features,'filtered_liwc.csv')
  # Save LIWC-based features from unfiltered data to 'unfiltered_liwc.csv'.
  save(unfiltered_liwc_features,'unfiltered_liwc.csv')

  # Save combined sentiment and word presence features from filtered data to 'filtered_combo.csv'.
  save(filtered_combo_features,'filtered_combo.csv')
  # Save combined sentiment and word presence features from unfiltered data to 'unfiltered_combo.csv'.
  save(unfiltered_combo_features,'unfiltered_combo.csv')
  


  # train classifier and show performance in cross-validation

  labels = [0,1,2,3,4]
  print("Cross Validation for all features(unfiltered) : \n ")

  print("\n Unigram Unfiltered : ")
  classifyKaggle_crossval.cross_validation_PRF(5,unfiltered_unigram_features,labels)
  print("\n Bigram Unfiltered : ")
  classifyKaggle_crossval.cross_validation_PRF(5,unfiltered_bigram_features,labels)
  print("\n Pos Unfiltered : ")
  classifyKaggle_crossval.cross_validation_PRF(5,unfiltered_pos_features,labels)
  print("\n SL Unfiltered : ")
  classifyKaggle_crossval.cross_validation_PRF(5,unfiltered_sl_features,labels)
  print("\n LIWC Unfiltered : ")
  classifyKaggle_crossval.cross_validation_PRF(5,unfiltered_liwc_features,labels)
  print("\n Combined SL LIWC Unfiltered : ")
  classifyKaggle_crossval.cross_validation_PRF(5,unfiltered_combo_features,labels)

  print("\n Unigram filtered : ")
  classifyKaggle_crossval.cross_validation_PRF(5,filtered_unigram_features,labels)
  print("\n Bigram filtered : ")
  classifyKaggle_crossval.cross_validation_PRF(5,filtered_bigram_features,labels)
  print("\n Pos filtered : ")
  classifyKaggle_crossval.cross_validation_PRF(5,filtered_pos_features,labels)
  print("\n SL filtered : ")
  classifyKaggle_crossval.cross_validation_PRF(5,filtered_sl_features,labels)
  print("\n LIWC filtered : ")
  classifyKaggle_crossval.cross_validation_PRF(5,filtered_liwc_features,labels)
  print("\n Combined SL LIWC filtered: ")
  classifyKaggle_crossval.cross_validation_PRF(5,filtered_combo_features,labels)



  print("\n Unigram Unfiltered : ")
  naivebayesaccuracy(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  naivebayesaccuracy(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  naivebayesaccuracy(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  naivebayesaccuracy(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  naivebayesaccuracy(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  naivebayesaccuracy(unfiltered_combo_features)


  print("\n Unigram filtered : ")
  naivebayesaccuracy(filtered_unigram_features)
  print("\n Bigram filtered : ")
  naivebayesaccuracy(filtered_bigram_features)
  print("\n Pos filtered : ")
  naivebayesaccuracy(filtered_pos_features)
  print("\n SL filtered : ")
  naivebayesaccuracy(filtered_sl_features)
  print("\n LIWC filtered : ")
  naivebayesaccuracy(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  naivebayesaccuracy(filtered_combo_features)

  print("--------------------------------------------------For desicion tree -----------------------------------------------")
  print("\n Unigram Unfiltered : ")
  dt(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  dt(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  dt(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  dt(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  dt(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  dt(unfiltered_combo_features)

  print("===== for filtered =====")


  print("\n Unigram filtered : ")
  dt(filtered_unigram_features)
  print("\n Bigram filtered : ")
  dt(filtered_bigram_features)
  print("\n Pos filtered : ")
  dt(filtered_pos_features)
  print("\n SL filtered : ")
  dt(filtered_sl_features)
  print("\n LIWC filtered : ")
  dt(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  dt(filtered_combo_features)


  



  print("--------------------------------------------------For svm -----------------------------------------------")
  print("\n Unigram Unfiltered : ")
  svm(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  svm(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  svm(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  svm(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  svm(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  svm(unfiltered_combo_features)

  print("===== for filtered =====")


  print("\n Unigram filtered : ")
  svm(filtered_unigram_features)
  print("\n Bigram filtered : ")
  svm(filtered_bigram_features)
  print("\n Pos filtered : ")
  svm(filtered_pos_features)
  print("\n SL filtered : ")
  svm(filtered_sl_features)
  print("\n LIWC filtered : ")
  svm(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  svm(filtered_combo_features)



  
  
 

  print("--------------------------------------------------For random forest-----------------------------------------------")
  print("\n Unigram Unfiltered : ")
  rf(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  rf(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  rf(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  rf(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  rf(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  rf(unfiltered_combo_features)

  print("===== for filtered =====")


  print("\n Unigram filtered : ")
  rf(filtered_unigram_features)
  print("\n Bigram filtered : ")
  rf(filtered_bigram_features)
  print("\n Pos filtered : ")
  rf(filtered_pos_features)
  print("\n SL filtered : ")
  rf(filtered_sl_features)
  print("\n LIWC filtered : ")
  rf(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  rf(filtered_combo_features)





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






