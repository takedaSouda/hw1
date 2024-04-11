import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import jieba

def zipf_law(corpus_dir):
    corpus_text = ""
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(corpus_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                corpus_text += file.read()

    words = list(jieba.cut(corpus_text))
    word_freq = Counter(words)
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    freq = [item[1] for item in sorted_word_freq]
    rank = np.arange(1, len(freq) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(np.log(rank), np.log(freq), marker='o', linestyle='', label='Actual Frequencies')
    max_freq = freq[0]
    expected_freq = [max_freq / r for r in rank]
    plt.plot(np.log(rank), np.log(expected_freq), linestyle='-', label="Zipf's Law")

    plt.title("Zipf's Law for Chinese Corpus")
    plt.xlabel('log(Rank)')
    plt.ylabel('log(Frequency)')
    plt.grid(True)
    plt.legend()
    plt.show()


corpus_dir = 'jyxstxtqj_downcc.com'
zipf_law(corpus_dir)