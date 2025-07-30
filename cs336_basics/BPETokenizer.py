import regex as re
from collections import defaultdict

class BPETokenizer:
    def __init__(self, merges_output_file: str, vocabulary_output_file: str):
        self.merges_output_file = merges_output_file
        self.vocabulary_output_file = vocabulary_output_file
        self.merges = []
        self.vocab = []

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

    def save_data(self):
        with open(self.merges_output_file, "w") as file:
            file.write(str(self.merges))

        with open(self.vocabulary_output_file, "w") as file:
            file.write(str(self.vocab))

    def train_bpe(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> (dict[int, bytes], list[tuple[bytes, bytes]]):
        # Add in special tokens to the vocabulary
        # Read the file on input_path
        file_object = open(input_path, "r")
        content = file_object.read()
        content_docs = re.split("|".join(map(re.escape, special_tokens)), content)

        merges: list[tuple[bytes, bytes]] = []
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        vocab_reverse: dict[bytes, int] = {bytes([i]): i for i in range(256)}

        for special_token in special_tokens:
            token_bytes = bytes(special_token, 'utf-8')
            if token_bytes not in vocab.values() and len(vocab.keys()) < vocab_size:
                vocab[len(vocab.keys())] = token_bytes

        counts: dict[bytes, int] = defaultdict(int)
        indices: dict[bytes, list[int]] = defaultdict(list)
        bytepairs_to_pretokens: dict[tuple[int, int], set[bytes]] = defaultdict(set)
        bytepairs_to_counts: dict[tuple[int, int], int] = defaultdict(int)

        for content_doc in content_docs:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for match in re.finditer(PAT, content_doc):
                pretoken = match.group().encode("utf-8")
                counts[pretoken] += 1
                indices[pretoken] = list(map(int, pretoken))
                for index1, index2 in zip(indices[pretoken], indices[pretoken][1:]):
                    bytepairs_to_pretokens[(index1, index2)].add(pretoken)
                    bytepairs_to_counts[(index1, index2)] += 1

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
                for index1, index2 in zip(indices[pretoken], indices[pretoken][1:]):
                    bytepairs_to_counts[(index1, index2)] -= counts[pretoken]
                    new_bytepairs_to_affected_pretokens[(index1, index2)] = set(bytepairs_to_pretokens[(index1, index2)])
                    new_bytepairs_to_affected_pretokens[(index1, index2)].remove(pretoken)

                new_indices = self.merge(indices[pretoken], pair, new_index)

                for index1, index2 in zip(new_indices, new_indices[1:]):
                    bytepairs_to_counts[(index1, index2)] += counts[pretoken]
                    if (index1, index2) not in new_bytepairs_to_affected_pretokens:
                        new_bytepairs_to_affected_pretokens[(index1, index2)] = set(bytepairs_to_pretokens[(index1, index2)])
                    new_bytepairs_to_affected_pretokens[(index1, index2)].add(pretoken)

                indices[pretoken] = new_indices

            bytepairs_to_pretokens[(index1, index2)] = set([])

            for bytepair in new_bytepairs_to_affected_pretokens:
                bytepairs_to_pretokens[bytepair] = new_bytepairs_to_affected_pretokens[bytepair]

            bytepairs_to_counts[pair] = 0

            i+=1

        self.merges = merges 
        self.vocab = vocab

        return (vocab, merges)

bpe_tokenizer = BPETokenizer('../data/tinystories_train_merges_output.txt', '../data/tinystories_train_vocab_output.txt')
bpe_tokenizer.train_bpe(input_path='../data/TinyStoriesV2-GPT4-train.txt', vocab_size=10000, special_tokens=["<|endoftext|>"])
bpe_tokenizer.save_data()
        
        