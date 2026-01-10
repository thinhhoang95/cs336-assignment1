def bytes_to_tuples(input: bytes) -> tuple(bytes):
    return tuple[bytes](bytes([b]) for b in input)

from collections import Counter 
def count_byte_pairs(input: tuple) -> dict(tuple[bytes, int]):
    f"""To return frequency of byte pairs

    Args:
        input (tuple): e.g., (b'e', b'm', b'y', b'e', b'u')

    Returns:
        dict: b'e': 2, b'm': 1, ...
    """
    byte_pair_counter = Counter()
    for i in range(len(input)-1):
        byte_pair = input[i], input[i+1]
        if byte_pair not in byte_pair_counter:
            byte_pair_counter[byte_pair] = 1
        else:
            byte_pair_counter[byte_pair] += 1
    return byte_pair_counter

def count_and_merge_byte_pairs(pretokens: list[tuple[bytes]], new_vocabs: int = 1):
    merges: list = [] 
    for vocab_to_add_index in range(new_vocabs):
        byte_pair_counter = Counter()
        for pretoken in pretokens:
            byte_pair_counter += count_byte_pairs(pretoken)

        # Find the most common byte pair
        most_frequent_byte_pair = max(byte_pair_counter.items(), key=lambda item: (item[1], item[0]))[0]

        merges.append(most_frequent_byte_pair)

        # Merge the two bytes in the most_frequent_byte_pair together and add to dictionary
        # To avoid directly mutating the pretokens list, we will mark deletions, note the corresponding replacements, then add them later
        tuples_to_remove = []
        tuples_to_append = []
        for pretoken in pretokens:
            for i in range(len(pretoken) - 1):
                if pretoken[i] == most_frequent_byte_pair[0] and pretoken[i+1] == most_frequent_byte_pair[1]:
                    # Take 0 -> i-1, most_frequent_byte_pair_merged
                    # Mark the pretoken for deletion
                    tuples_to_remove.append(pretoken)

                    tuple_replacement = pretoken[0:i] + (bytes(most_frequent_byte_pair[0] + most_frequent_byte_pair[1]),) + \
                        (pretoken[i+2:] if i + 2 < len(pretoken) else tuple())
                    tuples_to_append.append(tuple_replacement)

                    # For debugging
                    print(f'tuples_to_remove: {pretoken}')
                    print(f'tuples_to_append: {tuple_replacement}')

                    continue 

        for pretoken in tuples_to_remove:
            pretokens.remove(pretoken)

        for pretoken in tuples_to_append:
            pretokens.append(pretoken)

    return merges

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
    # 2. We create a list of tuples of bytes for each pretoken
    # 3. We run the merge procedure

def __test_bytes_to_tuples():
    print(bytes_to_tuples('êtao'.encode()))

def __test_count_bytepairs():
    print(count_byte_pairs('Я вас любил'))

def __test_merge_bytepairs():
    pretoks = ["Toi", "yeu", "em", "den", "nay", "chung", "co", "the",
        "ngon", "lua", "tinh", "chua", "han", "da", "tan", "phai"]

    pretok_bytes: list = [bytes_to_tuples(t.encode()) for t in pretoks]
    merges = count_and_merge_byte_pairs(pretok_bytes, new_vocabs=5)
    print(merges)

if __name__ == "__main__":
    __test_bytes_to_tuples()
    __test_count_bytepairs()
    __test_merge_bytepairs()