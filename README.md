# Project Documentation (for reviews)

## IPython Notebook Files

- `main.ipynb`: The main battle camp, the main scenario where I do all the data visualization, preprocessing and try everything that will make my model better.
- `alternative_scenarios.ipynb`: The answer to WHAT IF questions. In my case: what if I will do a different preprocessing or what if I will try and use another form of encoding. This document exists because I just didn't want to mess up the main with a lot of information.


## Python Files

- `utils.py`:  Utils contains all the functions that I am calling in the main, from showing some advanced plot to making some f1 score tables.


## Data Folder (`data/`)

- `input/`: Contains the .csv extracted from the original kaggle dataset (for pipelines and last chapter called Testing)
- `output/`: Contains submissions, the folder of 0 and 1, my model predictions towards `input/` files.
- `data.csv`: The original file provided by Sigmoid when they were sending the exam.
