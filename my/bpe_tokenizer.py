from pretokenizer import pretokenize_text
from rich.progress import track

def bytes_to_tuples(input: bytes) -> tuple(bytes):
    return tuple(bytes([b]) for b in input)

from collections import Counter 
def count_byte_pairs(input: tuple, n_occurrences: int = 1) -> dict(tuple[bytes, int]):
    f"""To return frequency of byte pairs. It also supports pretoken counts (optional).
    

    Args:
        input (tuple): e.g., (b'e', b'm', b'y', b'e', b'u')

    Returns:
        dict: b'e': 2 * n_occurrences, b'm': 1 * n_occurrences, ...
    """
    byte_pair_counter = Counter()
    for i in range(len(input)-1):
        byte_pair = input[i], input[i+1]
        if byte_pair not in byte_pair_counter:
            byte_pair_counter[byte_pair] = n_occurrences
        else:
            byte_pair_counter[byte_pair] += n_occurrences
    return byte_pair_counter

def scan_pretoken_and_merge(pretoken, most_frequent_byte_pair):
    i = 0
    original_pretoken = pretoken 
    while i < len(pretoken) - 1:
        if pretoken[i] == most_frequent_byte_pair[0] and pretoken[i+1] == most_frequent_byte_pair[1]:
            # Take 0 -> i-1, most_frequent_byte_pair_merged

            pretoken = pretoken[0:i] + (bytes(most_frequent_byte_pair[0] + most_frequent_byte_pair[1]),) + \
                (pretoken[i+2:] if i + 2 < len(pretoken) else tuple())

            i = 0 # reset the index so the scan could start from the beginning, so we could deal with
            # multiple occurrences of the merge 
        else:
            i += 1

    return original_pretoken, pretoken

def count_and_merge_byte_pairs(pretoken_counts: dict[tuple[bytes], int], new_vocabs: int = 1):
    merges: list = [] 
    for vocab_to_add_index in track(range(new_vocabs), description="Finding merges..."):
        byte_pair_counter = Counter()
        for pretoken in pretoken_counts:
            byte_pair_counter += count_byte_pairs(pretoken, pretoken_counts[pretoken])

        # Find the most common byte pair
        most_frequent_byte_pair = max(byte_pair_counter.items(), key=lambda item: (item[1], item[0]))[0]

        merges.append(most_frequent_byte_pair)

        # Merge the two bytes in the most_frequent_byte_pair together and add to dictionary
        # To avoid directly mutating the pretokens list, we will mark deletions, note the corresponding replacements, then add them later
        tuples_to_remove = []
        tuples_to_append = []
        n_occurrences_to_append = [] # storing the pretoken counts, because if we remove the key from the dict pretoken_counts, we lose the n_occurrences as well
        for pretoken in pretoken_counts:
            orig, mrged = scan_pretoken_and_merge(pretoken, most_frequent_byte_pair)
            tuples_to_remove.append(orig)
            tuples_to_append.append(mrged)

        for i, pretoken in enumerate(tuples_to_remove):
            n_occurrences_to_append.append(pretoken_counts[pretoken])
            del pretoken_counts[pretoken]
            # For robustness, always check to ensure n_occurrences_to_append and tuples_to_append do not stay out of sync in terms of index
            assert len(n_occurrences_to_append) == (i + 1)

        for i, pretoken in enumerate(tuples_to_append):
            if pretoken not in pretoken_counts:
                pretoken_counts[pretoken] = n_occurrences_to_append[i]
            else:
                pretoken_counts[pretoken] += n_occurrences_to_append[i]

    return merges

import os 
from pathlib import Path
import pickle

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    # 1. We run the pre-tokenizer, save the pretoken counts and load it
    # Create a random filename for the pretoken counts if not given in kwargs

    skip_pretokenization: bool = False

    if 'pretoken_counts_path' in kwargs:
        pretoken_counts_pathstr = kwargs['pretoken_counts_path']
        if os.path.exists(pretoken_counts_pathstr):
            print(f"Pretoken counts file {pretoken_counts_pathstr} exists. Skipping pretokenization step.")
            skip_pretokenization = True 
        else:
            print(f"Pretoken counts file {pretoken_counts_pathstr} does NOT exist. Will generate a new one.")
    else:
        from datetime import datetime
        pretoken_counts_pathstr = f'pretoken_counts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'

    if not skip_pretokenization:
            print(f'Pretokenizing corpus...')
            pretokenize_text(Path(input_path), special_tokens, Path(pretoken_counts_pathstr))

    print(f'Pretokenization completed.')
    
    with open(Path(pretoken_counts_pathstr), "rb") as f:
        pretoken_counts: dict[str, int] = pickle.load(f)
        print(f'Pretoken counts loaded from {pretoken_counts_pathstr}!')

    # 2. Prepare for the training task on the pretokens
    new_vocabs = vocab_size - 256 - len(special_tokens)

    print(f'New vocabularies: {new_vocabs}')
    
    # 3. Convert the pretoken_counts's keys to bytes
    # {"low": 4} -> {(b"l", b"o", b"w"): 4}
    pretok_bytes: dict[bytes, int] = dict() 
    for pretok in pretoken_counts:
        pretok_in_bytes = bytes_to_tuples(pretok.encode())
        pretok_bytes[pretok_in_bytes] = pretoken_counts[pretok]

    # 4. Run the merge process for `new_vocabs` times to merge the most frequent tokens together
    print(f'Training BPE...')
    merges = count_and_merge_byte_pairs(pretok_bytes, new_vocabs)

    print(f'Training completed.')
    print(merges[:5])


def __test_bytes_to_tuples():
    print(bytes_to_tuples('êtao'.encode()))

def __test_count_bytepairs():
    print(count_byte_pairs('Я вас любил'))


def __test_full_bpe_train():
    input_path = 'data/TinyStoriesV2-GPT4-valid.txt'
    special_tokens = ["<|endoftext|>"]
    new_vocabs = 256 + len(special_tokens) + 100
    run_train_bpe(input_path, new_vocabs, special_tokens, pretoken_counts_path = "output/pretoken_counts.pkl")

if __name__ == "__main__":
    # __test_bytes_to_tuples()
    # __test_count_bytepairs()
    __test_full_bpe_train()