# Movie Review Sentiment Classification

## Project Overview
This project focuses on the classification of sentiment in movie reviews, utilizing various natural language processing techniques. The goal is to accurately predict sentiment from textual data sourced from Rotten Tomatoes, as part of a Kaggle competition.

## Dataset
The dataset combines information from Socher et al.'s sentiment analysis research and Pang & Lee's movie review corpus. It includes multiple sentiment labels ranging from "Negative" to "Positive". The data is divided into training and testing sets, with labeled sentiments for training.

## Methodology
### Data Preprocessing
- **Punctuation Removal**: Utilized regular expressions to strip punctuation, reducing noise in the data.
- **Stop Word Filtering**: Removed common words that do not contribute to sentiment analysis using an expanded NLTK stopword list.
- **Tokenization and Lemmatization**: Applied tokenization to split text into words and lemmatization to reduce words to their base forms.
- **Bigrams**: Generated bigrams to capture word pair meanings not conveyed by single words.

### Feature Engineering
- **Bag of Words & Unigrams**: Created features based on word frequency and presence within the dataset.
- **Bigrams**: Used to capture the contextual relationship between adjacent words.
- **POS Tagging**: Analyzed the part of speech of words to improve model accuracy.
- **Sentiment Lexicon**: Incorporated sentiment lexicons to classify words based on their inherent sentiment.

### Experiments
Implemented various machine learning models including Naïve Bayes, Decision Tree, SVM, and Random Forest. Performed 5-fold cross-validation to evaluate model performance using metrics such as accuracy, precision, recall, and F1-Score.

## Results
Our models demonstrated varied effectiveness across different feature sets:
- **Unigrams and Bigrams** showed promising results, especially when unfiltered.
- **Filtered features** generally improved model performance in Decision Trees and Random Forests.
- **Sentiment Lexicon Features** and **POS Tagging** enhanced accuracy in specific scenarios.

## Challenges
Faced significant challenges in data preprocessing and integration of custom functions for feature extraction. Adjustments in the preprocessing steps were crucial for improving the input feature quality, leading to better model outcomes.

## Conclusion
The project highlighted the impact of feature selection and the robustness of different classifiers in sentiment analysis. The findings provide insights into the critical factors affecting sentiment classification accuracy and offer a foundation for future research.

