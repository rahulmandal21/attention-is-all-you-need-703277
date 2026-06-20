import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import List, Tuple

class DataPreprocessor:
    """
    A class used to preprocess data for the Transformer model.
    
    It includes tokenizing the input and output sequences, 
    converting the tokens into embeddings, and padding the sequences.
    """

    def __init__(self, min_freq: int = 2, max_len: int = 100):
        """
        Initializes the DataPreprocessor.
        
        Args:
        min_freq (int): The minimum frequency of a token to be included in the vocabulary.
        max_len (int): The maximum length of a sequence.
        """
        self.min_freq = min_freq
        self.max_len = max_len
        self.vocab = {}
        self.embedding = None

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text.
        
        Args:
        text (str): The input text.
        
        Returns:
        List[str]: A list of tokens.
        """
        return text.lower().split()

    def build_vocab(self, texts: List[str]) -> dict:
        """
        Builds the vocabulary from the input texts.
        
        Args:
        texts (List[str]): A list of input texts.
        
        Returns:
        dict: A dictionary mapping tokens to their indices.
        """
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))
        self.vocab = {
            word: idx + 1
            for idx, (word, count) in enumerate(counter.items())
            if count >= self.min_freq
        }
        self.vocab["<unk>"] = 0
        return self.vocab

    def encode(self, text: str) -> List[int]:
        """
        Encodes the input text into a list of token indices.
        
        Args:
        text (str): The input text.
        
        Returns:
        List[int]: A list of token indices.
        """
        return [self.vocab.get(tok, 0) for tok in self.tokenize(text)]

    def create_embedding(self, num_tokens: int, embedding_dim: int) -> nn.Embedding:
        """
        Creates an embedding layer.
        
        Args:
        num_tokens (int): The number of tokens in the vocabulary.
        embedding_dim (int): The dimension of the embedding space.
        
        Returns:
        nn.Embedding: The embedding layer.
        """
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        return self.embedding

    def pad_sequence(self, sequence: List[int]) -> torch.Tensor:
        """
        Pads the input sequence to the maximum length.
        
        Args:
        sequence (List[int]): The input sequence.
        
        Returns:
        torch.Tensor: The padded sequence.
        """
        padded_sequence = sequence + [0] * (self.max_len - len(sequence))
        return torch.tensor(padded_sequence)

class TokenizedTextDataset(Dataset):
    """
    A custom dataset class for tokenized text sequences.
    """

    def __init__(self, sequences: List[List[int]]):
        """
        Initializes the TokenizedTextDataset.
        
        Args:
        sequences (List[List[int]]): A list of tokenized sequences.
        """
        self.sequences = [torch.tensor(seq) for seq in sequences]

    def __len__(self) -> int:
        """
        Returns the number of sequences in the dataset.
        
        Returns:
        int: The number of sequences.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns the sequence at the given index.
        
        Args:
        idx (int): The index of the sequence.
        
        Returns:
        torch.Tensor: The sequence at the given index.
        """
        return self.sequences[idx]

def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    A custom collate function for padding sequences in a batch.
    
    Args:
    batch (List[torch.Tensor]): A list of sequences in the batch.
    
    Returns:
    torch.Tensor: The padded batch.
    """
    return pad_sequence(batch, batch_first=True, padding_value=0)

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    texts = ["This is a sample text", "Another sample text"]
    vocab = preprocessor.build_vocab(texts)
    encoded_texts = [preprocessor.encode(text) for text in texts]
    padded_texts = [preprocessor.pad_sequence(text) for text in encoded_texts]
    dataset = TokenizedTextDataset(encoded_texts)
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in data_loader:
        print(batch)