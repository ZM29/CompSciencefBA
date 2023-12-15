import random
import collections
import json
from typing import List
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


class FeaturesMap:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Product:
    def __init__(self, shop: str, url: str, modelID: str, featuresMap: dict, title: str):
        self.shop = shop
        self.url = url
        self.modelID = modelID
        self.featuresMap = FeaturesMap(**featuresMap)
        self.title = title


class ProductData:
    def __init__(self, productID: str, products: List[Product]):
        self.productID = productID
        self.products = products


def load_data(file_path: str) -> List[ProductData]:
    """
    Loads data from a JSON file (Path) and converts it into a list of ProductData objects containing 
    the product information.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    product_list = []
    for productID, products in data.items():
        product_objects = [Product(**product) for product in products]
        product_data = ProductData(productID=productID, products=product_objects)
        product_list.append(product_data)
    return product_list


def truth_and_prepare_data(productdata_list):
    """
    By using the list of ProductData objects, we return a tuple with the true data and prepared data
    """
    truth = collections.defaultdict(list)
    prepared_data = []
    for data in productdata_list:
        for product in data.products:
            # Extract true data
            truth[product.modelID].append(product)
            # Prepare data by discarding modelID
            product_without_modelID = Product(
                shop=product.shop,
                url=product.url,
                modelID=product.modelID,
                featuresMap=product.featuresMap.__dict__,
                title=product.title,
            )
            prepared_data.append(product_without_modelID)
    return truth, prepared_data


def preprocess_data(data):
    """
    Preprocesses product data by tokenizing text and creating tokenized representations. We return
    a list of dictionaries containing the tokenized representations of product information
    """
    preprocessed_data = []
    for product in data:
        # Combine product details into a single string
        string_text = f"{product.title} {' '.join([f'{k} {v}' for k, v in product.featuresMap.__dict__.items()])}"

        # Tokenize the text using NLTK's word tokenizer
        tokens = nltk.word_tokenize(string_text.lower())
        tokenized_text = ' '.join(tokens)

        preprocessed_data.append({product.modelID: tokenized_text})

    return preprocessed_data


def hash_functions_calculation(num_hash_functions):
    """
    Generate a list of hash functions.
    """
    def generate_hash_function():
        a = random.randint(1, 2**32 - 1)
        b = random.randint(0, 2**32 - 1)
        return lambda x: (a * x + b) % (2**32)

    return [generate_hash_function() for _ in range(num_hash_functions)]


def minhash_signature_calculation(tfidf_vector, hash_functions):
    """
    Computes the MinHash signature for the TF-IDF vector using a list of hash functions. We return
    the MinHash signature 
    """
    signature = [float('inf')] * len(hash_functions)
    non_zero = tfidf_vector.nonzero()[1]
    for index in non_zero:
        for i, hash_function in enumerate(hash_functions):
            hash_value = hash_function(index)
            if hash_value < signature[i]:
                signature[i] = hash_value
    return signature

def tfidf_vectors(preprocessed_data):
    """
    Generates TF-IDF vectors using the (list of dictionaries of) preprocessed data. We return a 
    tuple containing the TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    to_transform = [' '.join(product.values()) for product in preprocessed_data]
    tfidf_matrix = vectorizer.fit_transform(to_transform)
    return tfidf_matrix



def initialize_lsh(num_bands, rows_per_band):
    """
    Initializes an empty LSH structure with the given number of bands and rows per band.
    """
    lsh_structure = {
        'num_bands': num_bands,
        'rows_per_band': rows_per_band,
        'buckets': [{} for _ in range(num_bands)]
    }
    return lsh_structure


def apply_lsh(signatures, lsh_structure):
    """
    Applies LSH to the given signatures and populates the LSH structure.
    Each signature is split into bands and hashed into buckets within each band.
    """
    num_bands = lsh_structure['num_bands']
    rows_per_band = lsh_structure['rows_per_band']
    buckets = lsh_structure['buckets']

    for product in signatures:
        for model_id, signature in product.items():
            for band in range(num_bands):
                # Extract the part of the signature for this band
                start = band * rows_per_band
                end = start + rows_per_band
                band_signature = tuple(signature[start:end])
                
                # Hash the band signature to a bucket
                bucket_hash = hash(band_signature) % (2**32)
                
                # Add the model_id to the corresponding bucket
                if bucket_hash not in buckets[band]:
                    buckets[band][bucket_hash] = []
                buckets[band][bucket_hash].append(model_id)

    lsh_structure['buckets'] = buckets
    return lsh_structure


def candidate_pairs_search(lsh_structure):
    """
    Identifies candidate pairs that may be duplicates by examining the LSH buckets.
    """
    candidate_pairs = set()
    for band_buckets in lsh_structure['buckets']:
        for bucket in band_buckets.values():
            if len(bucket) > 1:
                # Sort the bucket to ensure consistent order of pairs
                sorted_bucket = sorted(bucket)
                for i in range(len(sorted_bucket)):
                    for j in range(i + 1, len(sorted_bucket)):
                        # Add the sorted pair to the set
                        candidate_pairs.add((sorted_bucket[i], sorted_bucket[j]))
    return candidate_pairs


def calculate_similarity(set1, set2):
    """
    Calculates the Jaccard similarity between two sets.
    """
    overlap = set1.intersection(set2)
    union = set1.union(set2)
    return len(overlap) / len(union)


def classify_duplicates(candidates, signatures, threshold):
    """
    Classifies candidate pairs as duplicates or non-duplicates based on the Jaccard similarity threshold.
    """
    duplicates = set()
    for candidate in candidates:
        set1 = None
        set2 = None
        for product in signatures:
            if candidate[0] in product:
                set1 = set(product[candidate[0]])
            if candidate[1] in product:
                set2 = set(product[candidate[1]])
            if set1 and set2:
                break
        if set1 and set2:
            similarity = calculate_similarity(set1, set2)
            if similarity >= threshold:
                duplicates.add(candidate)
    return duplicates


def bootstrap_sampling(dataset, num_bootstraps):
    """
    Performs bootstrapping on the dataset and separates training and test data based on a number of
    bootstraps. We return a list of tuples, where each tuple contains a training set and a test set.
    """
    bootstrap_samples = []
    n = len(dataset)
    train_size = int(n)
    
    for _ in range(num_bootstraps):
        # Create a bootstrap sample
        train_data = np.random.choice(dataset, train_size, replace=True)
        test_data = [instance for instance in dataset if instance not in train_data]
        bootstrap_samples.append((train_data, test_data))
    
    return bootstrap_samples


def get_duplicates(data):
    """
    Identifies and returns a list of actual duplicates in the data based on the productID or modelID.
    """
    duplicates = collections.defaultdict(list)
    for product in data:
        for product_id, _ in product.items():
            duplicates[product_id].append(product)
    return duplicates


def evaluate_performance(predictions, truth):
    """
    Evaluates and returns the performance of the duplicate detection algorithm using the F1-score.
    """
    TP = len([pair for pair in predictions if pair in truth])
    FP = len([pair for pair in predictions if pair not in truth])
    FN = len([pair for pair in truth if pair not in predictions])
    
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return f1_score


def calculate_pair_quality(duplicates, comparisons):
    """
    Calculates the pair quality metric.
    """
    pair_quality = len(duplicates) / comparisons if comparisons > 0 else 0
    return pair_quality


def calculate_pair_completeness(duplicates, total_duplicates):
    """
    Calculates the pair completeness metric.
    """
    pair_completeness = len(duplicates) / total_duplicates if total_duplicates > 0 else 0
    return pair_completeness


def calculate_f1_star(pair_quality, pair_completeness):
    """
    Calculates the F1* measure, which is the harmonic mean between pair quality and pair completeness.
    """
    if pair_quality + pair_completeness > 0:
        f1_star = 2 * (pair_quality * pair_completeness) / (pair_quality + pair_completeness)
    else:
        f1_star = 0
    return f1_star


def main():
    # Setting seed
    np.random.seed(42)

    # Load data
    file_path = 'data/data.json'
    product_list = load_data(file_path)

    # Extract truth and prepare data
    truth, prepared_data = truth_and_prepare_data(product_list)

    # Preprocess data
    preprocessed_data = preprocess_data(prepared_data)

    # Parameters for LSH, and similarity threshold
    num_bands = 2
    rows_per_band = 50
    similarity_threshold = (1 / num_bands) ** (1 / rows_per_band)
    print(similarity_threshold)
    num_hash_functions = 100  # Number of hash functions for MinHash

    # Generate hash functions for MinHash
    hash_functions = hash_functions_calculation(num_hash_functions)

    # Initialize LSH structure
    lsh_structure = initialize_lsh(num_bands, rows_per_band)

    # Bootstrap sampling and evaluation
    num_bootstraps = 5
    f1_scores = []
    pair_qualities = []
    pair_completenesses = []
    f1_stars = []
    fraction_comparisons = []

    bootstrap_samples = bootstrap_sampling(preprocessed_data, num_bootstraps)
    for i, (train_data, test_data) in enumerate(bootstrap_samples):
        print(f"Processing bootstrap sample {i + 1}...")

        # Generate TF-IDF vectors for the training data
        train_tfidf_matrix = tfidf_vectors(train_data)

        # Generate TF-IDF vectors for the test data
        test_tfidf_matrix = tfidf_vectors(test_data)

        # Generating MinHash signatures for each TF-IDF vector in the train set
        train_signatures = []
        for index in tqdm(range(len(train_data)), desc="Generating signatures for training"):
            tfidf_vector = train_tfidf_matrix[index]
            minhash_signature = minhash_signature_calculation(tfidf_vector, hash_functions)
            train_signatures.append({next(iter(train_data[index].keys())): minhash_signature})


        # Apply LSH to the training signatures
        print("Applying LSH to the training signatures...")
        train_lsh_structure = initialize_lsh(num_bands, rows_per_band)  # Re-initialize for training
        train_lsh_structure = apply_lsh(train_signatures, train_lsh_structure)

        # Generating MinHash signatures for each TF-IDF vector in the test set
        test_signatures = []
        for index in tqdm(range(len(test_data)), desc="Generating signatures for testing"):
            tfidf_vector = test_tfidf_matrix[index]
            minhash_signature = minhash_signature_calculation(tfidf_vector, hash_functions)
            test_signatures.append({next(iter(train_data[index].keys())): minhash_signature})

        # Find candidate pairs using LSH
        print("Finding candidate pairs using LSH...")
        test_candidate_pairs = candidate_pairs_search(train_lsh_structure)

        # Calculate the total number of possible comparisons between products in the test set
        length = len(test_data)
        total_comparisons = (length * (length - 1)) / 2  # All possible combinations 

        # Calculate the fraction of comparisons
        fraction = len(test_candidate_pairs) / total_comparisons
        fraction_comparisons.append(fraction)
        
        # Classify duplicates from candidate pairs
        print("Classifying duplicates from candidate pairs...")
        test_duplicates = classify_duplicates(test_candidate_pairs, test_signatures, similarity_threshold)
        
        # Get truth for the test set
        print("Getting truth for the test set...")
        truth_test = get_duplicates(test_data)
        truth_test_tuples = [(product, product) for product in truth_test]

        print(len(test_duplicates))
        print(len(truth_test_tuples))

        # Evaluate performance on the test set
        print("Evaluating performance on the test set...")
        f1_score = evaluate_performance(test_duplicates, truth_test_tuples)
        f1_scores.append(f1_score)

        # Calculate pair quality and completeness
        print("Calculating pair quality and completeness...")
        pair_quality = calculate_pair_quality(test_duplicates, len(test_candidate_pairs))
        pair_completeness = calculate_pair_completeness(test_duplicates, len(truth))
        pair_qualities.append(pair_quality)
        pair_completenesses.append(pair_completeness)

        # Calculate F1* measure
        print("Calculating F1* measure...")
        f1_star = calculate_f1_star(pair_quality, pair_completeness)
        f1_stars.append(f1_star)

    
    # Calculate average metrics across bootstraps
    print("Calculating average metrics across bootstraps...")
    avg_f1_score = np.mean(f1_scores)
    avg_pair_quality = np.mean(pair_qualities)
    avg_pair_completeness = np.mean(pair_completenesses)
    avg_f1_star = np.mean(f1_stars)
    avg_fraction_comparisons = np.mean(fraction_comparisons)
    
    # Output the evaluation results
    print("Outputting the evaluation results...")
    print(f"Average F1-score: {avg_f1_score}")
    print(f"Average Pair Quality: {avg_pair_quality}")
    print(f"Average Pair Completeness: {avg_pair_completeness}")
    print(f"Average F1* measure: {avg_f1_star}")
    print(f"Average Fraction of Comparisons: {avg_fraction_comparisons}")


if __name__ == '__main__':
    main()