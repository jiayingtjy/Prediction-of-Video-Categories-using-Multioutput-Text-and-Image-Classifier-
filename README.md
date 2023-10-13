# Prediction of Video Categories using Multioutput Text and Image Classifier 
 This project involves building a text classification model to predict video categories based on the video's title and transcript, along with an image classification model for video categories using thumbnails, providing a comprehensive analysis of video categorization.
## For detailed justification of steps and data pipeline please refer to the jupyter notebook or the pdf version of the notebook.
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

# PART 2: Image Classification

For this image classification report, I have gone through the whole data pipeline from data understanding, preparation, exploratory data analysis, modelling, evaluation, prediction and finally comparison of models. To optimize the performance of the model, I have tried many methodologies and switched the order of the data preparations and modelling preparation steps especially for image processing. The insights derived from a wide variety of models also contributed greatly to the overall objective, which is to classify video category based on the thumbnail image data. It is evident that different image processing steps and different models with different parameters can indeed affect model performance. Additionally, the more models tested, the higher probability of finding a good model for our predictions. I hope this report will provide great value to you and assist you to find the best model that will derive the most accurate and precise predictions in predicting the video category based on thumbnail image. Thank you.

### Basic Data Preprocessing
This section involves the initial steps in data preparation. We create a reference dataframe to link filenames to channels and categories, establish an image dataframe to store image pixels and video categories, and merge the two dataframes for further analysis.

### Data Understanding & Exploratory Data Analysis
A deeper exploration of the dataset is conducted in this section, where i have used techniques such as Histogram Gradient analysis to analyse the images 

### Data Preparation
This section justifies the selection of data preparation techniques and documents the steps used for image processing.After much exploratory data analysis and understanding of the data from different classes, I will moving on the data preparation of the images. After much trial and error, I realised that <b>identifying the correct processing steps is the most useful for increasing the model performance.</b> Hence, I will be documenting in this section, the selection of data preparation techniques to be used and justifying why based on factors such as model processing power and model perfomance.

| Data Preparation Step                                     | f1-score | Include Step? | Rationale                                           |
| -------------------------------------------------------- | -------- | ------------- | --------------------------------------------------- |
| Resize - (256, 256, 3)                                  | 0.454    | Yes           | Compulsory to prevent ResourceExhaustError          |
| Resize - (256, 256, 3) + Gamma Correction               | 0.454    | No            | No impact on model performance and may affect image properties |
| Resize - (256, 256, 3) + Bilateral Blurring             | 0.461    | Yes           | Improves model performance by 0.1%                    |
| Resize - (256, 256, 3) + Bilateral Blurring + Image Denoising | 0.454    | No            | Model performance decreases by 1%                     |
| Resize - (256, 256, 3) + Bilateral Blurring + Histogram Equalization | 0.454    | No            | Model performance decreases by 1%                     |
| Resize - (256, 256, 3) + Bilateral Blurring + Histogram of Oriented Gradient (HOG) | 0.46     | No            | Model performance decreases by 0.01%                   |
| Resize - (256, 256, 3) + Bilateral Blurring + Canny Edge Detection | 0.46     | No            | Model performance decreases by 0.01%                   |
| Resize - (256, 256, 3) + Bilateral Blurring + Grayscaling | 0.47     | Yes           | Model performance increases by 0.2%                    |
| Resize - (256, 256, 3) + Bilateral Blurring + VGG Feature Extraction | 0.61     | Yes           | Model performance increases by 16%                     |
| Resize - (256, 256, 3) + Bilateral Blurring + VGG Feature Extraction + Gamma Correction (value = 1.1) | 0.62     | Yes           | Model performance increases by 17%                     |


### Modelling

In this section, we explore different machine learning models for image classification. We tested various combinations of data preparation steps and models, fine-tuning parameters to achieve the best results. Here is a summary of the model performances:

| Model                              | Model Information                                          | Accuracy | Precision | Recall | F1-Score |
| ---------------------------------- | ---------------------------------------------------------- | -------- | --------- | ------ | -------- |
| Random Forest Classifier            | MultiOutputClassifier(RandomForestClassifier(n_estimators=1000)) | 0.27     | 0.65      | 0.27   | 0.23     |
| Logistic Regression                 | MultiOutputClassifier(estimator=LogisticRegression(C=100, penalty="none")) | 0.64     | 0.75      | 0.64   | 0.65     |
| Multinomial Naive Bayes             | MultiOutputClassifier(estimator=MultinomialNB(alpha=0.0, fit_prior=false)) | 0.56     | 0.63      | 0.56   | 0.55     |
| Support Vector Machine (SVM)        | MultiOutputClassifier(LinearSVC(C=9, dual=false))         | 0.62     | 0.74      | 0.62   | 0.64     |

These results highlight the performance of different machine learning models. The F1-score, which takes into account both precision and recall, is the primary metric of focus due to the multilabel classification nature of the problem.Fine-tuning model parameters allowed us to identify the best-performing model. The results of each model provide insights into their respective strengths and weaknesses in the context of image classification.
For detailed analysis, including the impact of data preparation, please refer to the notebook. Further experimentation and optimization may help to enhance model performance and achieve better results in image classification.

### Comparison of Predictions from text and image classifier
### Comparison of Models

This section provides an overall comparison of the best-performing models in both Task 1 (Text Classification) and Task 2 (Image Classification) for predicting video categories. The comparison is based on their respective fine-tuned models.

| Task  | Model                               | Model Information                                          | Accuracy | Precision | Recall | F1-Score |
| ----- | ----------------------------------- | ---------------------------------------------------------- | -------- | --------- | ------ | -------- |
| Task 1 | Text Classification (Tf-idf + Logistic Regression) | Pipeline([('vect', TfidfVectorizer(max_df=0.8, max_features=10000, ngram_range=(1,2)), ('clf', MultiOutputClassifier(LogisticRegression(class_weight='balanced', max_iter=3000)))]) | 0.83     | 0.94      | 0.88   | 0.91     |
| Task 2 | Image Classification (Logistic Regression) | MultiOutputClassifier(estimator=LogisticRegression(C=100, penalty="none")) | 0.64     | 0.75      | 0.64   | 0.65     |

### Insights and Analysis:

In this comparison, the best-performing models from both Task 1 and Task 2 are highlighted. Notably, both of these models are based on Logistic Regression. However, the key difference is the type of data used for prediction.

- **Task 1 (Text Classification)**: The model achieved an F1-Score of 0.91, making it highly effective in classifying video categories based on text data (video titles and transcripts).

- **Task 2 (Image Classification)**: The model achieved an F1-Score of 0.65, indicating decent performance. However, it lags behind the text classification model.

This section reflects on the findings and analysis from both Task 1 and Task 2. It highlights the differences in model performance and draws recommendations for future work. Specifically, it suggests focusing more on the video transcript and title for building recommender systems, or acquiring additional high-quality image data to enhance the model's accuracy.

The comprehensive exploration of image classification in this project can be a valuable resource for improving video categorization based on thumbnail images. It also provides important insights into the challenges of image classification compared to text classification.


