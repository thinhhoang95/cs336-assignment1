import os
from typing import BinaryIO
from collections import Counter
import multiprocessing
import pickle 

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import regex as re
def process_story(text: str, pretokens_dict: dict[bytes, int] = None):
    if pretokens_dict is None:
        pretokens_dict: dict[bytes, int] = dict()

    matches = re.finditer(PRETOKENIZATION_PATTERN, text)
    for match in matches:
        pretok = match.group() # for example: "I" or "'ll" or "love"
        if pretok not in pretokens_dict:
            pretokens_dict[pretok] = 1
        else:
            pretokens_dict[pretok] += 1

    return Counter(pretokens_dict)

def process_chunk(chunk_args):
    start = chunk_args[0]
    end = chunk_args[1]
    text_corpus_path = chunk_args[2]

    with open(text_corpus_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # print(f"Text chunk: {chunk[:20]} ... {chunk[-20:]} | Length: {len(chunk)}")

        # Split the stories at the special <|endoftext|> token 
        stories = re.split(r"<\|endoftext\|>", chunk) # match any blank spaces before and after the endoftext token. Note we need to escape the "OR" operator | with a backslash
        chunk_pretoken_counts = Counter()
        for story in stories:
            # print(f"Text story: {story[:20]} ... {story[-20:]} | Length {len(story)}")
            # Run pre-tokenization on the story and store the counts for each pre-token
            pretoken_counts = process_story(story, None)
            chunk_pretoken_counts += pretoken_counts

    return chunk_pretoken_counts

if __name__ == "__main__":
    ## Usage
    text_corpus_path = "data/TinyStoriesV2-GPT4-valid.txt"
    num_processes = 4
    with open(text_corpus_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_args = [(boundaries[i], boundaries[i+1], text_corpus_path) for i in range(len(boundaries)-1)] # [(0, 255122), (255122, 353124), ...]

    with multiprocessing.Pool(processes=num_processes) as pool:
        chunk_pretoken_counts = pool.map(process_chunk, chunk_args)

    # REDUCE Step: to assimilate all chunk_pretoken_counts into a master count object
    from functools import reduce
    total_counts = reduce(lambda x, y: x + y, chunk_pretoken_counts)

    # Show the first 5 counts
    print_count = 0
    for pretok in total_counts:
        print(f"{pretok} → {total_counts[pretok]}")
        print_count += 1
        if print_count == 5:
            break

    # Write the final results to a file
    with open("pretoken_counts.pkl", "wb") as f_prime:
        pickle.dump(total_counts, f_prime)
        print("Total counts saved to pretoken_counts.pkl")

