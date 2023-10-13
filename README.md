# Prediction of Video Categories using Multioutput Text and Image Classifier 
 This project involves building a text classification model to predict video categories based on the video's title and transcript, along with an image classification model for video categories using thumbnails, providing a comprehensive analysis of video categorization.
# PART 1: Text Classification
Certainly, let's provide a concise summary for each section, with a focus on the model results:

### Overview

This project encompasses the entire data pipeline, from data understanding and preparation to modeling and model comparison. The primary objective is to classify video categories based on titles and transcripts, exploring various text representations and models to optimize performance.

### Basic Data Understanding

In this section, I gained a fundamental understanding of the dataset. It was observed that the classification task is multilabel due to overlapping categories.

### Data Preparation
Data preparation involved handling textual columns ("Title" and "Transcript") and the "Category" column.
I also did text representation. It aims to numerically represent the unstructured text documents to make them mathematically computable.
### Text Representation

For text representation, three main techniques were employed:

1. **Bag of Words (BOW):** This approach converts text data into numerical feature vectors by counting the frequency of each word in the text. BOW was used in combination with various classification models.

2. **Term Frequency-Inverse Document Frequency (Tf-idf):** Tf-idf represents the importance of a term in a document relative to a collection of documents. It was applied in conjunction with different classification models.

3. **Word2Vec (W2V):** Word2Vec represents words as continuous vectors in a multi-dimensional space, capturing semantic relationships. Word2Vec was used for feature extraction alongside Logistic Regression and Support Vector Machine.

These text representations played a crucial role in shaping the model results and performance in the classification task.

### Data Understanding - Exploratory Data Analysis

Exploratory Data Analysis provided insights into descriptive statistics, video category analysis, and textual analysis across different video categories.

### Modeling

The modeling section explored multiple combinations of text representations and classification models, including Logistic Regression, Multinomial Naive Bayes, and Support Vector Machine (SVM).

### Model Results

**Logistic Regression Models:**
- BOW + Logistic Regression achieved an accuracy of 0.80, precision of 0.90, recall of 0.88, and an F1-score of 0.89.
- Tf-idf + Logistic Regression achieved an accuracy of 0.83, precision of 0.94, recall of 0.88, and an F1-score of 0.91.
- Word2Vec + Logistic Regression achieved an accuracy of 0.66, precision of 0.79, recall of 0.89, and an F1-score of 0.84.

**Multinomial Naive Bayes Models:**
- BOW + Multinomial Naive Bayes achieved an accuracy of 0.62, precision of 0.74, recall of 0.95, and an F1-score of 0.82.
- Tf-idf + Multinomial Naive Bayes achieved an accuracy of 0.69, precision of 0.92, recall of 0.75, and an F1-score of 0.81.

**Support Vector Machine Models:**
- BOW + SVM achieved an accuracy of 0.71, precision of 0.87, recall of 0.81, and an F1-score of 0.84.
- Tf-idf + SVM achieved an accuracy of 0.82, precision of 0.95, recall of 0.85, and an F1-score of 0.90.
- Word2Vec + SVM achieved an accuracy of 0.70, precision of 0.86, recall of 0.80, and an F1-score of 0.82.

### Comparison of Models

This section provided a comprehensive comparison of all model results, highlighting the best-performing models for video category classification.

### Conclusion

The conclusion section summarized the key findings and insights from the project, with an emphasis on selecting the most suitable model for video category prediction.
