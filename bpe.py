"""Implements the byte-pair encoding algorithm in pure Python.

Well, we implement "character-pair" encoding so we don't have to deal with conversion to bytes
and back. This is OK for this toy implementation because we construct the starting alphabet from 
all characters that occur in our dataset and we always test "in-distribution." In practice this 
approach may potentially break for unseen words.

Tokenization usually consists of a pipeline of 3 stages:
1. Normalization -> e.g. lowercasing, stemming, removing punctuation, etc.
2. Pre-tokenization -> e.g. whitespace tokenization into coarse words, newlines, punctuation, etc.
3. Sub-word tokenization -> e.g. BPE. 

Following GPT-2, we don't do any normalization (stage 1) and do pre-tokenization (stage 2) using
the same process as GPT-2.

Finally, this is a straightforward implementation mostly meant to understand the algorithm, and 
therefore highly inefficient.
"""

import regex as re
import argparse
from copy import deepcopy
from collections import defaultdict

from tqdm import tqdm

# GPT-2 pattern, might be overkill for our tinyshakespeare text.
PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _get_subwords(words: list[str]) -> list[list[str]]:
    return [[c for c in word] for word in words]


def _merge_subword(subword: list[str], merge: tuple[str, str]) -> list[str]:
    merged = []
    i = 0
    while i < len(subword):
        if i < len(subword) - 1 and (subword[i], subword[i + 1]) == merge:
            merged.append("".join([subword[i], subword[i+1]]))
            i += 2
        else:
            merged.append(subword[i])
            i += 1
    return merged


def construct_merges(words: list[str], n_merges: int) -> list[tuple[str, str]]:
    subwords = _get_subwords(words)
    merges = []
    for _ in tqdm(range(n_merges), desc="Constructing merges"):
        # Find the most frequently occurring bigram.
        bigram_counts = defaultdict(int)
        max_bigram, max_count = None, 0
        for subword in subwords:
            for l, r in zip(subword[:-1], subword[1:]):
                bigram = (l, r)
                bigram_counts[bigram] += 1
                if bigram_counts[bigram] > max_count:
                    max_bigram, max_count = bigram, bigram_counts[bigram]

        # Return if no more merges otherwise append to merges list.
        if max_bigram is None:
            print("No more merges possible, returning current list.")
            break
        merges.append(max_bigram)
        
        # Merge bigrams across all subwords.
        for i, subword in enumerate(subwords):
            last_merge = merges[-1]
            subwords[i] = _merge_subword(subword, last_merge)

    return merges


def tokenize(words: list[str], merges: list[tuple[str, str]]) -> list[str]:
    subwords = _get_subwords(words)
    tokens = []
    for subword in subwords:
        for merge in merges:
            subword = _merge_subword(subword, merge)
        tokens.extend(subword)
    return tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a simple character-pair tokenizer.")
    parser.add_argument("--num-merges", type=int, default=25)
    args = parser.parse_args()

    data = open("data/tinyshakespeare.txt").read()

    words = PATTERN.findall(data)
    merges = construct_merges(words, n_merges=args.num_merges)
    
    test_string = "Hello, I am Eugen Hotaj.\nThis is my BPE implementation\nI really hope you like it!"
    test_words = PATTERN.findall(test_string)
    num_base_tokens = sum([len(subword) for subword in _get_subwords(test_words)])
    tokens = tokenize(test_words, merges)
    print("Tokenized text:", tokens)
    print(f"Tokenizer efficiency: {num_base_tokens / len(tokens):.5}")
