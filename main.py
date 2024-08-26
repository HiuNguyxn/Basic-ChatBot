import numpy as np
from vncorenlp import VnCoreNLP
import py_vncorenlp
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel


# Import thư viện cần thiết
from underthesea import word_tokenize

# Hàm tách từ bằng underthesea
def tokenize(text):
    # Tách từ và trả về danh sách các từ
    tokens = word_tokenize(text)
    return tokens

# Hàm Bag of Words
def bag_of_word(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
