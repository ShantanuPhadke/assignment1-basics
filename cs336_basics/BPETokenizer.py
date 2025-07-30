import regex as re
from collections import defaultdict
from typing import BinaryIO
import os
import time
import multiprocessing

class BPETokenizer:
    def __init__(self, merges_output_file: str = 'default_merges_file.txt', vocabulary_output_file: str = 'default_vocabulary_file.txt'):
        self.merges_output_file = merges_output_file
        self.vocabulary_output_file = vocabulary_output_file
        self.merges = []
        self.vocab = []
        self.num_processes = 10

    def merge(self, indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
        """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
        new_indices = []  # @inspect new_indices
        i = 0  # @inspect i
        while i < len(indices):
            if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
                new_indices.append(new_index)
                i += 2
            else:
                new_indices.append(indices[i])
                i += 1
        return new_indices

    def find_chunk_boundaries(
        self,
        file: BinaryIO, 
        desired_num_chunks: int, 
        split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

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

    def save_data(self):
        with open(self.merges_output_file, "w") as file:
            file.write(str(self.merges))

        with open(self.vocabulary_output_file, "w") as file:
            file.write(str(self.vocab))

    def generate_pretoken_counts(self, chunk):
        counts: dict[bytes, int] = defaultdict(int)
        indices: dict[bytes, list[int]] = defaultdict(list)

        content_docs = re.split("|".join(self.special_tokens), chunk)

        for content_doc in content_docs:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for match in re.finditer(PAT, content_doc):
                pretoken = match.group().encode("utf-8")
                counts[pretoken] += 1
                indices[pretoken] = list(map(int, pretoken))

        return (counts, indices)

    def combine_pretoken_results(self, results):
        combined_counts: dict[bytes, int] = defaultdict(int)
        combined_indices: dict[bytes, list[int]] = defaultdict(list)
        for result in results:
            counts, indices = result
            for pretoken in counts:
                combined_counts[pretoken] += counts[pretoken]
                if pretoken not in combined_indices:
                    combined_indices[pretoken] = indices[pretoken]
        return (combined_counts, combined_indices)


    def train_bpe(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> (dict[int, bytes], list[tuple[bytes, bytes]]):
        # Add in special tokens to the vocabulary
        # Read the file on input_path
        self.special_tokens = special_tokens

        merges: list[tuple[bytes, bytes]] = []
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        vocab_reverse: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        chunks = []

        for special_token in special_tokens:
            token_bytes = bytes(special_token, 'utf-8')
            if len(vocab.keys()) < vocab_size:
                vocab[len(vocab.keys())] = token_bytes

        with open(input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(
                f, self.num_processes, "<|endoftext|>".encode("utf-8")
            )
            
            # The following is a serial implementation, but you can parallelize this 
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                results = pool.map(self.generate_pretoken_counts, chunks)
                combined_counts, combined_indices = self.combine_pretoken_results(results)

                bytepairs_to_pretokens: dict[tuple[int, int], set[bytes]] = defaultdict(set)
                bytepairs_to_counts: dict[tuple[int, int], int] = defaultdict(int)

                for pretoken in combined_counts:
                    pretoken_count = combined_counts[pretoken]
                    for index1, index2 in zip(combined_indices[pretoken], combined_indices[pretoken][1:]):
                        bytepairs_to_pretokens[(index1, index2)].add(pretoken)
                        bytepairs_to_counts[(index1, index2)] += pretoken_count

                num_merges = vocab_size - len(vocab.keys())

                i = 0

                while i < num_merges and max(bytepairs_to_counts.values()) > 1:
                    pair = max(bytepairs_to_counts, key=lambda k: (bytepairs_to_counts.get(k), (vocab[k[0]], vocab[k[1]])))
                    index1, index2 = pair
                    new_index = len(vocab.keys())
                    vocab[new_index] = vocab[index1] + vocab[index2]
                    merges.append((vocab[index1], vocab[index2]))
                    affected_pretokens = bytepairs_to_pretokens[(index1, index2)]

                    new_bytepairs_to_affected_pretokens = defaultdict(set)
                    for pretoken in affected_pretokens:
                        pretoken_count = combined_counts[pretoken]
                        for index1, index2 in zip(combined_indices[pretoken], combined_indices[pretoken][1:]):
                            bytepairs_to_counts[(index1, index2)] -= pretoken_count
                            new_bytepairs_to_affected_pretokens[(index1, index2)] = set(bytepairs_to_pretokens[(index1, index2)])
                            new_bytepairs_to_affected_pretokens[(index1, index2)].remove(pretoken)

                        new_indices = self.merge(combined_indices[pretoken], pair, new_index)

                        for index1, index2 in zip(new_indices, new_indices[1:]):
                            bytepairs_to_counts[(index1, index2)] += pretoken_count
                            if (index1, index2) not in new_bytepairs_to_affected_pretokens:
                                new_bytepairs_to_affected_pretokens[(index1, index2)] = bytepairs_to_pretokens[(index1, index2)]
                            new_bytepairs_to_affected_pretokens[(index1, index2)].add(pretoken)

                        combined_indices[pretoken] = new_indices

                    bytepairs_to_pretokens[(index1, index2)] = set([])

                    for bytepair in new_bytepairs_to_affected_pretokens:
                        bytepairs_to_pretokens[bytepair] = new_bytepairs_to_affected_pretokens[bytepair]

                    bytepairs_to_counts[pair] = 0

                    i+=1

        self.merges = merges 
        self.vocab = vocab

        return (vocab, merges)

'''
if __name__ == "__main__":
    start_time = time.time()
    bpe_tokenizer = BPETokenizer('../data/tinystories_train_merges_output.txt', '../data/tinystories_train_vocab_output.txt')
    bpe_tokenizer.train_bpe(input_path='../data/TinyStoriesV2-GPT4-train.txt', vocab_size=10000, special_tokens=["<|endoftext|>"])
    end_time = time.time()
    print('Total Time = ' + str(end_time - start_time))
    bpe_tokenizer.save_data()
'''
        
        