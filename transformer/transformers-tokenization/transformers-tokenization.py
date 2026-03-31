import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        unique = set()
        
        for text in texts:
            for word in text.split():
                unique.add(word)
        
        unique = sorted(unique)
        
        special = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for idx, token in enumerate(special):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        for idx, word in enumerate(unique, start=4):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

        self.vocab_size = len(self.word_to_id)
            
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        res = []
        unk_id = self.word_to_id[self.unk_token]
        words = text.split()
        for word in words:
            id = self.word_to_id.get(word)            
            if id is None:
                res.append(unk_id)
            else:
                res.append(id)
        
        return res
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        res = []
        for i in ids:
            word = self.id_to_word.get(i)
            if word is None:
                res.append(self.unk_token)
            
            else:
                res.append(word)
                
        return " ".join(res)
                
