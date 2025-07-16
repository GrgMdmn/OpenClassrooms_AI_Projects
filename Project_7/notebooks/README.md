# Notebooks

This folder contains the Jupyter notebooks used for model development and analyses in the project.

## Contents

- Exploratory Data Analysis
- Classical Models (logistic regression, SVM, etc.)
- Advanced Models (LSTM with GloVe embeddings)
- BERT Model

## Note on requirements files

The `*requirements.txt` files were generated from a Google Colab environment. This environment includes many pre-installed libraries, not all necessary for the project.  

Manual cleanup or specific generation of requirements files will be needed to slim down these files before production use or deployment.

---

## Model Performance Summary

### Classical Models (trained on 200,000 tweets)

| Model                        | Precision | Training Time   |
|------------------------------|-----------|-----------------|
| Random Forest + BoW          | 0.7798    | ~5 hours        |
| Logistic Regression + TF-IDF | 0.7796    | <2 minutes      |

### Advanced Models based on word-embeddings (trained on 200,000 tweets, GPU accelerated on Google Colab)

| Model                  | Precision | Training Time |
|------------------------|-----------|---------------|
| Word2Vec + LSTM        | 0.797     | 57 seconds    |
| GloVe (300d) + LSTM    | 0.792     | 47 seconds    |

### State-of-the-art Model (trained on 100,000 tweets due to RAM limits)

| Model  | Precision | Training Time |
|--------|-----------|---------------|
| BERT   | 0.882     | 1.3 minutes   |

---

This summary highlights the trade-offs between model precision and training time. While the classical models are faster to train, advanced models leveraging embeddings and LSTM yield better precision, with BERT achieving the highest precision despite being trained on a smaller dataset.

The precision offset between BERT and word embeddings models may mainly be due to the BERT bidirectionnal context abilities.
