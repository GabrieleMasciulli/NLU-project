import torch
import torch.utils.data as data
import numpy as np
import os
import requests
import zipfile
from tqdm import tqdm

DEVICE = torch.device("cuda:0")

# Loading the corpus


def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Vocab with tokens to ids


def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

# This class computes and stores our vocab
# Word to ids and ids to word


class Lang:
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}


class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            # We get from the first token till the second-last token
            self.source.append(sentence.split()[0:-1])
            # We get from the second token till the last token
            self.target.append(sentence.split()[1:])
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    # Map sequences of tokens to corresponding computed in Lang class
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    print('You have to deal with that')
                    break
            res.append(tmp_seq)
        return res


def load_glove_embeddings(path, word2id, emb_dim):
    """
    Loads GloVe embeddings from a file and creates a weight matrix
    for the words present in the provided vocabulary.
    """
    print(f"Loading GloVe embeddings from {path}...")
    embeddings_index = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                # Use the first word as the key
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print(f"Error: GloVe file not found at {path}.")
        print("Please download GloVe vectors (e.g., glove.6B.zip from https://nlp.stanford.edu/projects/glove/)")
        print("and place the extracted .txt file (e.g., glove.6B.300d.txt) in the correct directory.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the GloVe file: {e}")
        return None

    vocab_size = len(word2id)
    # Initialize embedding matrix with small random values using float32 for compatibility with PyTorch default tensor type
    embedding_matrix = np.random.randn(
        vocab_size, emb_dim).astype(np.float32) * 0.01

    found_words = 0
    for word, i in word2id.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Check if the loaded vector dimension matches the expected emb_dim
            if len(embedding_vector) == emb_dim:
                embedding_matrix[i] = embedding_vector
                found_words += 1
            else:
                print(
                    f"Warning: Dimension mismatch for word '{word}'. Expected {emb_dim}, got {len(embedding_vector)}. Skipping.")
        # Handle special tokens like <pad> or <eos> if they are not in GloVe
        # They will retain their random initialization, which is often fine.
        # Alternatively, you could initialize them to zeros or averages.

    print(f"Loaded {len(embeddings_index)} word vectors.")
    print(
        f"Found {found_words}/{vocab_size} words from vocabulary in GloVe file.")

    # Ensure the matrix is float32
    return torch.tensor(embedding_matrix, dtype=torch.float32)


def download_and_extract_glove(url, download_dir, expected_filename):
    """Downloads and extracts the specified GloVe file if it doesn't exist."""
    os.makedirs(download_dir, exist_ok=True)
    glove_txt_path = os.path.join(download_dir, expected_filename)

    if os.path.exists(glove_txt_path):
        print(
            f"GloVe file '{expected_filename}' already exists in '{download_dir}'. Skipping download.")
        return glove_txt_path

    zip_filename = url.split('/')[-1]
    zip_filepath = os.path.join(download_dir, zip_filename)

    print(f"Downloading GloVe embeddings from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192  # 8KB
            progress_bar = tqdm(total=total_size, unit='iB',
                                unit_scale=True, desc=f"Downloading {zip_filename}")
            with open(zip_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
            if total_size != 0 and progress_bar.n != total_size:
                print("ERROR, something went wrong during download")
                return None

        print(f"Extracting '{expected_filename}' from {zip_filename}...")
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            # Ensure the specific file exists in the archive before extracting
            if expected_filename in zip_ref.namelist():
                zip_ref.extract(expected_filename, path=download_dir)
                print(f"Successfully extracted '{expected_filename}'.")
            else:
                print(
                    f"Error: '{expected_filename}' not found in the downloaded zip file.")
                return None

    except requests.exceptions.RequestException as e:
        print(f"Error downloading GloVe embeddings: {e}")
        return None
    except zipfile.BadZipFile:
        print(
            f"Error: Downloaded file '{zip_filename}' is not a valid zip file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        # Clean up the zip file
        if os.path.exists(zip_filepath):
            print(f"Removing downloaded zip file: {zip_filepath}")
            os.remove(zip_filepath)

    return glove_txt_path
