# Sentiment Analysis
johnsons 16c dictionary sentiment analysis tool

## conda env set-up

```
/path/to/python -m pip install ipykernel
/path/to/python -m ipykernel install --user
pip install -U "tensorflow-text==2.18.1"
pip install "tf-models-official==2.18.0"
```

## HPC job submission script


## Bert Regression Workflow
This workflow uses DistilBERT to learn sentiment scores from manually annotated entries in Johnson's Dictionary and then predict sentiment for unseen dictionary entries. I inputted my csv which contained 3 categories where: 
word = Johnson headword
def = dictionary definition
sentiment = manually assigned sentiment score (-3 to +3)

Install required packages:
pip install pandas scikit-learn transformers datasets torch accelerate

To run the model execute: 

python bert-regression.py

The script will: Load sentiment_scored.csv, convert sentiment values to numeric format, remove missing values, split data into training and test sets (80/20), tokenise definitions using DistilBERT, fine-tune a regression model, evaluate performance using Mean Absolute Error (MAE), and output example predictions.

Example:
"Full of affection; strongly moved; warm; zealous."
→ 3
"To put to pain; to grieve; to torment."
→ -3

Performance is measured using:
Mean Absolute Error (MAE)
Formula: MAE = average absolute difference between predicted and actual score

Interpretation:
MAE = 0.0 Perfect prediction
MAE = 0.5 Very strong
MAE = 1.0 Reasonable
MAE > 1.5 Weak

Current result:
Training examples: 520
Test examples: 131
MAE ≈ 0.92
This indicates that the model is able to learn meaningful relationships between Johnson's definitions and manually assigned sentiment scores.

Output
Example predictions:
True: -3.0 Predicted: -2.35
True: 2.0 Predicted: 1.78
True: -2.75 Predicted: -2.39

Notes: Automatically generated sentiment values were excluded from training.
DistilBERT was trained as a regression model, predicting continuous sentiment values rather than positive/negative categories.
The sentiment scale ranges from:
-3 = strongly negative
0 = neutral
+3 = strongly positive

This approach aims to produce a historically informed sentiment lexicon suitable for eighteenth-century texts and subsequent application to large historical newspaper collections.
