import pandas as pd
from collections import Counter
import math

def sequence2series(sequence):
    ch2id = {}
    count = 1
    for chr in sequence:
        if chr not in ch2id:
            ch2id[chr] = count
            count += 1

    transfered_seq = []
    for chr in sequence:
        transfered_seq.append(ch2id[chr])

    return transfered_seq

def series2sequence(series, num_bin):
    time_series = pd.Series(series)

    labels = []
    for i in range(num_bin):
        labels.append(str(i))
    # Discretize into equal-width bins
    bins = pd.cut(time_series, bins=num_bin, labels=labels)

    return bins.to_list()

def poitwise_mutual_information(words1, words2):
    assert len(words1) == len(words2), "Sentences must be of equal length"

    # Count occurrences of individual words and matching aligned word pairs
    word_counts1 = Counter(words1)
    word_counts2 = Counter(words2)
    pair_counts = Counter((word1, word2) for word1, word2 in zip(words1, words2) if word1 == word2)
    total_pairs = len(words1)

    # Calculate Normalized Mutual Information (NMI) between aligned word pairs where words are the same
    nmi_values = []
    for word1, word2 in zip(words1, words2):
        if word1 == word2:  # Only calculate NMI if the words match
            # Probability of each word in the individual sentences
            p_x = word_counts1[word1] / total_pairs
            p_y = word_counts2[word2] / total_pairs
            # Joint probability of the identical word pair
            p_xy = pair_counts[(word1, word2)] / total_pairs

            # Calculate NMI for the word pair
            if p_xy > 0:  # Avoid log(0)
                mi = math.log(p_xy / (p_x * p_y), 2)
                nmi = mi / -math.log(p_xy, 2)  # Normalize MI
                nmi_values.append(nmi)
        else:
            # If words are not the same, we assume zero mutual information for that pair
            nmi_values.append(0)

    # Aggregate NMI values (e.g., by averaging) to get a single score
    average_nmi = sum(nmi_values) / len(nmi_values) if nmi_values else 0
    return average_nmi


if __name__ == "__main__":
    series = [0.1, 0.5, 0.9, 1.4, 2.5, 3.1, 5.0, 7.2, 8.0]
    print(series)
    seq = series2sequence(series, 3)
    print(seq)

    s = sequence2series(seq)
    print(s)

    sentence1 = "the cat plays outside".split(' ')
    sentence2 = "the dog plays outside".split(' ')

    pmi = poitwise_mutual_information(sentence1, sentence2)

    print(pmi)