import regex as re
from collections import defaultdict
from typing import BinaryIO
import os
import time
import multiprocessing
import ast
from collections.abc import Iterable, Iterator
from cs336_basics.CodeProfiler import CodeProfiler

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
        pass

    def calculate_encoding(self, pretokens:list[list[bytes]]) -> list[int]:
        pass

    def encode(self, text: str) -> list[int]:
        encoding = []

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # self.vocab = {b' ': 0, b'a': 1, b'c': 2, b'e': 3, b'h': 4, b't': 5, b'th': 6, b' c': 7, b' a': 8, b'the': 9, b' at': 10 }
        # self.merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]

        vocab_reverse = {}
        for v in self.vocab:
            vocab_reverse[self.vocab[v]] = v

        print('text = ' + text)
        print()

        content_docs = [text]
        special_tokens_ordering = []

        if self.special_tokens:
            # Sort special tokens from longest to shortest
            print('special tokens before sorting = ' + str(self.special_tokens))
            self.special_tokens.sort(key=lambda token: -1*len(token))
            # content doc index -> index -> special token
            # starting_indices_to_special_token = {}
            print('special tokens after sorting = ' + str(self.special_tokens))
            #content_docs = re.split("|".join(self.special_tokens), text)
            for special_token in self.special_tokens:
                special_token_length = len(special_token)
                content_doc_index = 0
                for content_doc in content_docs:
                    #if content_doc_index not in starting_indices_to_special_token:
                    #    starting_indices_to_special_token[content_doc_index] = {}
                    for i in range(len(content_doc) - special_token_length+1):
                        if content_doc[i:i+special_token_length] == special_token:
                            special_tokens_ordering.append(special_token)
                            # Take out this chunk of text so we don't double count it for overlapping tokens
                            content_docs[content_doc_index] = content_doc[:i] + content_doc[i+special_token_length:]
                    content_doc_index += 1
                print('special_tokens_ordering = ' + str(special_tokens_ordering))
                print('content_docs = ' + str(content_docs))
                print()

            content_docs = re.split("|".join([re.escape(spt) for spt in self.special_tokens]), text)

        print('Processing content_docs = ' + str(content_docs))

        processed_content_docs = []
        special_token_index = 0
        #special_token_match_started = False
        for content_doc in content_docs:
            for match in re.finditer(PAT, content_doc):
                # Whenever '' is encountered its a special token
                if match.group() == '':
                    #if special_token_match_started:
                    processed_content_docs.append(self.special_tokens[0].encode('utf-8'))
                    #special_token_match_started = False
                    #else:
                    #    special_token_match_started = True
                else:
                    processed_content_docs.append(match.group())
            # Every content doc is split at a special token, find the correct special token and add the encoded version in
            if special_token_index < len(special_tokens_ordering):
                processed_content_docs.append(special_tokens_ordering[special_token_index].encode('utf-8'))
                special_token_index+=1

        pretokens = []
        for processed_content in processed_content_docs:
            if isinstance(processed_content, bytes):
                pretokens.append([processed_content])
            else:
                pretokens.append([processed_content.encode('utf-8')[i:i+1] for i in range(len(processed_content.encode('utf-8')))])

        # For each pretoken, keep doing the following till you can't find a matching merge anymore:
        #   (1) Find the first matching merge
        #   (2) Apply the matching merge

        print('pretokens pre-processing = ' + str(pretokens))
        for pretoken_index in range(len(pretokens)):
            pretoken = pretokens[pretoken_index]
            print('pretoken pre-processing = ' + str(pretoken))
            merge_index = 0
            can_merge = len(pretoken) > 1
            while merge_index < len(self.merges) and can_merge:
                merge = self.merges[merge_index]
                if merge[0] in pretoken and merge[1] in pretoken[pretoken.index(merge[0])+1:]:
                    print('Potentially matching merge detected = (' + str(merge[0]) + ', ' + str(merge[1]) + ')')
                    possible_merge0_indices = []
                    possible_merge1_indices = []
                    for l1 in range(len(pretoken)):
                        if pretoken[l1] == merge[0]:
                            possible_merge0_indices.append(l1)
                    for l2 in range(len(pretoken)):
                        if pretoken[l2] == merge[1]:
                            possible_merge1_indices.append(l2)
                    l1_pointer = 0
                    l2_pointer = 0
                    merge_not_found = True
                    merges_found = []
                    while l1_pointer < len(possible_merge0_indices) and l2_pointer < len(possible_merge1_indices):
                        l1_index = possible_merge0_indices[l1_pointer]
                        l2_index = possible_merge1_indices[l2_pointer]
                        if l2_index - l1_index == 1:
                            merge_not_found = False
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
                        print('Merged! Pretoken afterwords = ' + str(pretoken))

                if len(pretoken) == 1:
                    can_merge = False
                merge_index+=1

            print('pretoken post-processing = ' + str(pretoken))
            print()

            encoding.extend([vocab_reverse[b] for b in pretoken])

            #[b' Univers', b'i', b'ty'] -> (b' Univers', b'i'), (b'i', b'ty')
            # Example 'theta' -> [t, h, e, t, a]
            # Merge found at beginning: (t, h) -> new bytepairs= [t, h, e, t, a] -> [th, h, e, t, a] -> pop(h) -> [th, e, t, a]
            # Merge found at end: (t, a) -> new bytepairs=[t, h, e, t, a] -> [t, h, e, ta, a] -> pop(a) -> [t, h, e, ta]
            # Merge found in the middle: (h, e) -> new bytepairs=[t, h, e, t, a] -> [t, he, e, t, a] -> pop(e)

            # Multi-merging examples:
            # Example ' the' -> (t, h), (h, e)
            # Merge 1: (th, e)
            # Merge 2: (the)                  

        #print('overall ecoding = ' + str(encoding))
        return encoding

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        bytestr = b''
        for vocab_index in ids:
            bytestr += self.vocab[vocab_index]
        print('bytesr_raw = ' + str(bytestr))
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


'''
if __name__ == "__main__":
    bpe_tokenizer = BPETokenizer('../data/tinystories_train_merges_output.txt', '../data/tinystories_train_vocab_output.txt')
    bpe_tokenizer.train_bpe(input_path='../data/TinyStoriesV2-GPT4-train.txt', vocab_size=10000, special_tokens=["<|endoftext|>"])
    bpe_tokenizer.save_data()
    print(bpe_tokenizer.code_profiler)
'''

#bpe_tokenizer = BPETokenizer.from_files(BPETokenizer, '../data/tinystories_train_vocab_output.txt', '../data/tinystories_train_merges_output.txt', special_tokens=["<|endoftext|>"])
#bpe_tokenizer.encode('the cat ate')
        
        