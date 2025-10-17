# Detecting Fake News with TF-IDF and Naive Bayes (NB)

Authors: Joshua Harrison, Fiona Tan, Walid Eid, Claudia Scott

This project looks to find optimal features to use when detecting COVID-19 fake news from Twitter/X posts, focusing on a simple, speedy method of extracting features and classifying it as "Fake" or not based on the contents of the post. This could easily be scaled to perform real time analysis of tweets through the Twitter/X API if required (X Developer Platform, n.d.)

This project uses Sci-kit Learn's implementation of Multinomial Naive Bayes for the NB calculations in the code.

## How to use:

1. Clone the repository and navigate into it.
2. Create a virtual environment: `python3 -m venv venv`
3. Activate the environment: `source venv/bin/activate` (linux/mac), `source venv/bin/activate.ps1` (windows from powershell)
4. Install requirements: `pip install -r requirements.txt`
5. Run the code: `python3 main.py`

Results will be outputted to a text file called `results.txt`.

The code in `demo.py` is used for presentation purposes.

See `pseudocode.md` for a simple summary of the experimentation algorithm used by the `main.py` program.

## Sample Results:

```
0
Experiment: 0x0
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.27961	 Model Accuracy: 0.6611570247933884
Confusion Matrix:
              Predicted False  Predicted True
Actual False               11              31
Actual True                10              69

1
Experiment: 0x1
Removing Hashtags
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.33101	 Model Accuracy: 0.6942148760330579
Confusion Matrix:
              Predicted False  Predicted True
Actual False               11              31
Actual True                 6              73

2
Experiment: 0x10
Using length
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.07331	 Model Accuracy: 0.6363636363636364
Confusion Matrix:
              Predicted False  Predicted True
Actual False               17              25
Actual True                19              60

3
Experiment: 0x11
Removing Hashtags
Using length
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.08211	 Model Accuracy: 0.6694214876033058
Confusion Matrix:
              Predicted False  Predicted True
Actual False               16              26
Actual True                14              65

4
Experiment: 0x100
Using average word length
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.12001	 Model Accuracy: 0.6528925619834711
Confusion Matrix:
              Predicted False  Predicted True
Actual False               14              28
Actual True                14              65

5
Experiment: 0x101
Removing Hashtags
Using average word length
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.21361000000000002	 Model Accuracy: 0.6528925619834711
Confusion Matrix:
              Predicted False  Predicted True
Actual False                4              38
Actual True                 4              75

6
Experiment: 0x110
Using length
Using average word length
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.07951	 Model Accuracy: 0.6363636363636364
Confusion Matrix:
              Predicted False  Predicted True
Actual False               16              26
Actual True                18              61

7
Experiment: 0x111
Removing Hashtags
Using length
Using average word length
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.07961	 Model Accuracy: 0.6528925619834711
Confusion Matrix:
              Predicted False  Predicted True
Actual False               15              27
Actual True                15              64

8
Experiment: 0x1000
Using readability score
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.18551	 Model Accuracy: 0.6611570247933884
Confusion Matrix:
              Predicted False  Predicted True
Actual False               14              28
Actual True                13              66

9
Experiment: 0x1001
Removing Hashtags
Using readability score
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.37721000000000005	 Model Accuracy: 0.6776859504132231
Confusion Matrix:
              Predicted False  Predicted True
Actual False                7              35
Actual True                 4              75

10
Experiment: 0x1010
Using length
Using readability score
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.07581	 Model Accuracy: 0.628099173553719
Confusion Matrix:
              Predicted False  Predicted True
Actual False               16              26
Actual True                19              60

11
Experiment: 0x1011
Removing Hashtags
Using length
Using readability score
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.08741	 Model Accuracy: 0.6776859504132231
Confusion Matrix:
              Predicted False  Predicted True
Actual False               15              27
Actual True                12              67

12
Experiment: 0x1100
Using average word length
Using readability score
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.16971000000000003	 Model Accuracy: 0.628099173553719
Confusion Matrix:
              Predicted False  Predicted True
Actual False                7              35
Actual True                10              69

13
Experiment: 0x1101
Removing Hashtags
Using average word length
Using readability score
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.19741000000000003	 Model Accuracy: 0.6611570247933884
Confusion Matrix:
              Predicted False  Predicted True
Actual False                6              36
Actual True                 5              74

14
Experiment: 0x1110
Using length
Using average word length
Using readability score
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.08461	 Model Accuracy: 0.6446280991735537
Confusion Matrix:
              Predicted False  Predicted True
Actual False               16              26
Actual True                17              62

15
Experiment: 0x1111
Removing Hashtags
Using length
Using average word length
Using readability score
Fitting 5 folds for each of 50000 candidates, totalling 250000 fits
Alpha: 0.08291	 Model Accuracy: 0.6363636363636364
Confusion Matrix:
              Predicted False  Predicted True
Actual False               13              29
Actual True                15              64

Best test: 1. Highest accuracy: 0.6942148760330579
```

## References
X Developer Platform (n.d.). *X API*. https://developer.x.com/en/docs/x-api. Accessed 17/10/2025.