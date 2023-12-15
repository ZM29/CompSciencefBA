# Duplication Detection With LSH and TF-IDF

This project aims to detect duplicates within a dataset containing product information using TF-IDF vectors and Locality-Sensitive Hashing (LSH).

## Overview

The code implements a method to identify potential duplicate products within a dataset by employing the following steps:

1. **Data Preprocessing**: Extraction and Tokenization of product information using NLTK's word tokenizer and creation of TF-IDF vectors using `TfidfVectorizer` from `sklearn`.

2. **MinHash Signature Generation**: Creation of MinHash signatures for TF-IDF vectors to approximate Jaccard similarity between products.

3. **Locality-Sensitive Hashing (LSH)**: Application of LSH to MinHash signatures for efficient identification of candidate duplicate pairs.

4. **Evaluation and Metrics**: Calculation of performance metrics including F1-score, pair quality, pair completeness, and F1* measure across bootstraps to assess the accuracy of duplicate detection.

## Code Structure

The codebase is structured as follows:

- `CSfBA_TF_IDF.py`: The main logic for data loading, preprocessing, MinHash computation, LSH application, and evaluation.
- `data/`: Folder containing the dataset (`data.json`) used for duplicate detection.
- `README.md`: This file, providing an overview of the project, its structure, and usage instructions.

## How to Use

To run the code:

1. Clone the repository: https://github.com/ZM29/CompSciencefBA.git

2. Install dependencies (assuming Python and pip are already installed):
    ```bash
    pip install -r requirements.txt
    ```

3. Run the main script:
    ```bash
    python CSfBA_TF_IDF.py
    ```

## Dataset

The provided `data.json` file contains product information structured as a JSON object, where each key represents a unique ID and the corresponding value is a list of products with similar features.
