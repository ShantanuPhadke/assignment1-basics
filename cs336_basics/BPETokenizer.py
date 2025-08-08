import regex as re
from collections import defaultdict
from typing import BinaryIO
import os
import time
import multiprocessing
import ast
from collections.abc import Iterable, Iterator
from cs336_basics.CodeProfiler import CodeProfiler
from cs336_basics.BPEIterator import BPEIterator

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes] = dict([]), merges: list[tuple[bytes, bytes]] = [], special_tokens: list[str] | None = None, merges_output_file: str = 'default_merges_file.txt', vocabulary_output_file: str = 'default_vocabulary_file.txt'):
        self.merges_output_file = merges_output_file
        self.vocabulary_output_file = vocabulary_output_file
        self.merges = merges
        self.vocab = vocab
        self.special_tokens = special_tokens
        self.num_processes = 4
        self.code_profiler = CodeProfiler()

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

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = []
        merges = []
        with open(vocab_filepath, "r") as f:
            vocab_str = f.read()
            vocab = ast.literal_eval(vocab_str)

        with open(merges_filepath, "r") as f:
            merges_str = f.read()
            merges = ast.literal_eval(merges_str)

        return BPETokenizer(
            vocab=vocab, merges=merges, special_tokens=special_tokens,
        )


    def calculate_pretokens(self, text:str) -> list[list[bytes]]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        content_docs = [text]
        special_tokens_ordering = []

        if self.special_tokens:
            # Sort special tokens from longest to shortest to handle overlapping tokens
            self.special_tokens.sort(key=lambda token: -1*len(token))

            for special_token in self.special_tokens:
                special_token_length = len(special_token)
                content_doc_index = 0
                for content_doc in content_docs:
                    for i in range(len(content_doc) - special_token_length+1):
                        if content_doc[i:i+special_token_length] == special_token:
                            special_tokens_ordering.append(special_token)
                            # Take out this chunk of text so we don't double count it for overlapping tokens
                            content_docs[content_doc_index] = content_doc[:i] + content_doc[i+special_token_length:]
                    content_doc_index += 1

            content_docs = re.split("|".join([re.escape(spt) for spt in self.special_tokens]), text)

        processed_content_docs = []
        special_token_index = 0

        for content_doc in content_docs:
            for match in re.finditer(PAT, content_doc):
                processed_content_docs.append(match.group())
            # Every content doc is split at a special token unless we run out of matches, find the correct special token and add the encoded version in
            if special_token_index < len(special_tokens_ordering):
                processed_content_docs.append(special_tokens_ordering[special_token_index].encode('utf-8'))
                special_token_index+=1

        pretokens = []
        for processed_content in processed_content_docs:
            if isinstance(processed_content, bytes):
                pretokens.append([processed_content])
            else:
                pretokens.append([processed_content.encode('utf-8')[i:i+1] for i in range(len(processed_content.encode('utf-8')))])

        return pretokens

    def encode(self, text: str) -> list[int]:
        encoding = []

        # For each pretoken, keep doing the following till you can't find a matching merge anymore:
        #   (1) Find the first matching merge
        #   (2) Apply the matching merge

        pretokens = self.calculate_pretokens(text)

        vocab_reverse = {}
        for v in self.vocab:
            vocab_reverse[self.vocab[v]] = v

        for pretoken_index in range(len(pretokens)):
            pretoken = pretokens[pretoken_index]
            merge_index = 0
            can_merge = len(pretoken) > 1
            while merge_index < len(self.merges) and can_merge:
                merge = self.merges[merge_index]
                if merge[0] in pretoken and merge[1] in pretoken[pretoken.index(merge[0])+1:]:
                    # These are lists because sometimes we may have to merge multiple instances of a bytepair in a pretoken
                    possible_merge0_indices = []
                    possible_merge1_indices = []

                    l1 = 0
                    while l1 < len(pretoken)-1:
                        if pretoken[l1] == merge[0]:
                            if pretoken[l1+1] == merge[1]:
                                possible_merge0_indices.append(l1)
                                possible_merge1_indices.append(l1+1)
                                l1+=1
                        l1+=1

                    l1_pointer = 0
                    l2_pointer = 0
                    merges_found = []
                    while l1_pointer < len(possible_merge0_indices) and l2_pointer < len(possible_merge1_indices):
                        l1_index = possible_merge0_indices[l1_pointer]
                        l2_index = possible_merge1_indices[l2_pointer]
                        if l2_index - l1_index == 1:
                            merges_found.append((l1_index, l2_index))
                            l1_pointer+=1
                            l2_pointer+=1
                        elif l2_index - l1_index < 1:
                            l2_pointer+=1
                        else:
                            l1_pointer+=1

                    if len(merges_found) > 0:
                        for merge_found in merges_found[::-1]:
                            l1_index, l2_index = merge_found
                            pretoken[l1_index] += pretoken[l2_index]
                            pretoken.pop(l2_index)

                if len(pretoken) == 1:
                    can_merge = False
                merge_index+=1

            encoding.extend([vocab_reverse[b] for b in pretoken])              

        return encoding

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return BPEIterator(my_tokenizer=self, my_iterable=iterable)

    def decode(self, ids: list[int]) -> str:
        bytestr = b''
        for vocab_index in ids:
            bytestr += self.vocab[vocab_index]
        return bytestr.decode('utf-8', errors='replace')

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

        self.code_profiler.start_new_profiler(name='Pretokenization')

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

                self.code_profiler.log_profiler()

                self.code_profiler.start_new_profiler(name='Bytepair Merging')

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

                self.code_profiler.log_profiler()

        self.merges = merges 
        self.vocab = vocab

        return (vocab, merges)
