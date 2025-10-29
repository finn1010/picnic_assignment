# Text Classification Task
Selected dataset: AG News

## 1. Data Used
Two text classification models trained to categorise text into four topics:
1. World
2. Sports
3. Business
4. Sci/Tech

Preprocessing:
- Lowercasing
- Removing punctuation, digits, HTML artifacts, URLs, News Tags
- Tokenizer handled by TfidfVectorizer

## 2. Models Used
Two models were trained using TF-IDF features
- Logistic Regression (sklearn.linear_model.LogisticRegression)
- Multinomial Naive Bayes (sklearn.naive_bayes.MultinomialNB)

### Feature extraction
TfidfVectorizer(max_features=5000, ngram_range=(1,2))

## 3. Results
Logistic Regression (TF-IDF) | Acc: 0.903 | F1: 0.903

Naive Bayes (TF-IDF)         | Acc: 0.892 | F1: 0.892

Logistic Regression performs better than Naive Bayes, however, the difference is trivial. This suggests that TF-IDF features dominate the overall performance. This aligns with findings suggesting both typically misclassify the same samples (error correlation: 0.76), implying overlapping decision boundaries. Hence, engineering new features would likely result in stronger improvements to model performance in comparison to using a LR + NB ensemble. Further work on feature engieering might involve  expanding n gram ranges, TF-IDF smoothing or integrating embedding-based representations. These methodological adjsutments combined with hyperparameter tuning could improve classification performance. 

