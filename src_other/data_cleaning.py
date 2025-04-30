### SOURCE: https://github.com/iamsuvhro/IMDB-Reviews-Classification/blob/master/Data_cleaning.ipynb ###
import os
import re
import time
import multiprocessing
import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from kaggle.api.kaggle_api_extended import KaggleApi

# Download stopwords
nltk.download('stopwords', quiet = True)

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = ToktokTokenizer()
        self.stopword_list = stopwords.words('english')

    def clean_text(self, text):
        """Applies all text cleaning functions to the input text."""
        text = self.lowercase(text)
        text = self.remove_html(text)
        text = self.remove_url(text)
        text = self.remove_punctuation(text)
        text = self.remove_extra_characters(text)
        text = self.remove_stopwords(text)
        text = self.stem_text(text)
        return text

    def lowercase(self, text):
        replacements = {
            ",000,000": " m", ",000": " k", "′": "'", "’": "'", 
            "won't": "will not", "cannot": "can not", "can't": "can not",
            "n't": " not", "what's": "what is", "it's": "it is",
            "'ve": " have", "'m": " am", "'re": " are", 
            "he's": "he is", "she's": "she is", "'s": " own",
            "%": " percent ", "₹": " rupee ", "$": " dollar ", "€": " euro ",
            "'ll": " will", "how's": "how has", "y'all": "you all",
            "o'clock": "of the clock", "ne'er": "never", "let's": "let us",
            "finna": "fixing to", "gonna": "going to", "gimme": "give me",
            "gotta": "got to", "'d": " would", "daresn't": "dare not",
            "dasn't": "dare not", "e'er": "ever", "everyone's": "everyone is",
            "'cause'": "because"
        }
        text = str(text).lower()
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r"([0-9]+)000000", r"\1m", text)
        text = re.sub(r"([0-9]+)000", r"\1k", text)
        return text

    def remove_html(self, text):
        return re.sub('<.*?>', ' ', text)

    def remove_url(self, text):
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"www.\S+", " ", text)
        return text

    def remove_punctuation(self, text):
        return re.sub('[^a-zA-Z]', ' ', text)

    def remove_extra_characters(self, text):
        return re.sub("\s*\b(?=\w*(\w)\1{2,})\w*\b", ' ', text)

    def remove_stopwords(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = [token for token in tokens if token.lower() not in self.stopword_list]
        return ' '.join(tokens)

    def stem_text(self, text):
        stemmer = nltk.PorterStemmer()
        return ' '.join([stemmer.stem(word) for word in text.split()])


class DatasetDownloader:
    def __init__(self, dataset_name, output_dir):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(self):
        """Download and unzip dataset from Kaggle."""
        self.api.dataset_download_files(self.dataset_name, path=self.output_dir, unzip=True)
        return pd.read_csv(os.path.join(self.output_dir, 'IMDB Dataset.csv'))


class Word2VecTrainer:
    def __init__(self, vector_size=100, window=2, min_count=20, epochs=35):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def train_model(self, tokenized_texts):
        cores = multiprocessing.cpu_count()
        self.model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sample=6e-5,
            alpha=0.03,
            min_alpha=0.0007,
            negative=20,
            workers=cores - 1
        )
        start_time = time.time()
        self.model.build_vocab(tokenized_texts, progress_per=10000)
        print('Time to build vocab: {:.2f} mins'.format((time.time() - start_time) / 60))

        start_time = time.time()
        self.model.train(tokenized_texts, total_examples=self.model.corpus_count, epochs=self.epochs, report_delay=1)
        print('Time to train the model: {:.2f} mins'.format((time.time() - start_time) / 60))

    def convert_review_to_vectors(self, review):
        words = review.split()
        vectors = [self.model.wv[word] if word in self.model.wv else np.zeros(self.vector_size) for word in words]
        return np.array(vectors)

    def review_to_average_vector(self, review_vectors):
        return np.mean(review_vectors, axis=0) if len(review_vectors) > 0 else np.zeros(self.vector_size)


class DataPreparer:
    def __init__(self, output_dir = "datasets", max_length=300):
        self.output_dir = output_dir
        self.max_length = max_length

    def pad_review_vectors(self, review_vector):
        if len(review_vector) < self.max_length:
            padding = np.zeros(self.max_length - len(review_vector))
            return np.concatenate((review_vector, padding))
        return review_vector[:self.max_length]

    def save_processed_data(self, train, test):
        train.to_csv(os.path.join(self.output_dir, 'reviews_train.csv'), index=False)
        test.to_csv(os.path.join(self.output_dir, 'reviews_test.csv'), index=False)


class DataCleaningPipeline:
    def __init__(self, output_dir = "datasets", dataset_name = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"):
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.text_processor = TextPreprocessor()
        self.downloader = DatasetDownloader(dataset_name, output_dir)
        self.word2vec_trainer = Word2VecTrainer()
        self.data_preparer = DataPreparer(output_dir)

    def process_and_save_data(self):
        df = self.downloader.download_dataset()
        
        # Split dataset and preprocess text
        train, test = train_test_split(df, test_size=0.5, stratify=df["sentiment"], random_state=42)
        train["review"] = train["review"].apply(self.text_processor.clean_text)
        test["review"] = test["review"].apply(self.text_processor.clean_text)
        train['sentiment'] = train['sentiment'].map({'positive': 1, 'negative': 0})
        test['sentiment'] = test['sentiment'].map({'positive': 1, 'negative': 0})

        # Train Word2Vec
        tokenized_reviews = pd.concat([train, test], axis=0)["review"].apply(lambda x: x.split())
        self.word2vec_trainer.train_model(tokenized_reviews)

        # Convert reviews to average vectors, pad and save
        train["review"] = train["review"].apply(lambda x: self.word2vec_trainer.convert_review_to_vectors(x))
        test["review"] = test["review"].apply(lambda x: self.word2vec_trainer.convert_review_to_vectors(x))
        # Convert reviews to average vectors
        train["review"] = train["review"].apply(self.word2vec_trainer.review_to_average_vector)
        test["review"] = test["review"].apply(self.word2vec_trainer.review_to_average_vector)
        # Convert reviews to padded vectors
        train["review"] = train["review"].apply(self.data_preparer.pad_review_vectors)
        test["review"] = test["review"].apply(self.data_preparer.pad_review_vectors)

        self.data_preparer.save_processed_data(train, test)


# Usage
# output_dir = "datasets"
# dataset_name = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
# pipeline = DataCleaningPipeline(output_dir, dataset_name)
# pipeline.process_and_save_data()

# download_split_clean_dataset("/Users/manuel/Desktop/uni/Tesi magistrale/Githubs/FFA-Mine/datasets")
# x = pd.read_csv("/Users/manuel/Desktop/uni/Tesi magistrale/Githubs/FFA-Mine/datasets/reviews_train.csv")
# import torch
# train_data = x['review'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32))
# train_data = torch.stack([torch.tensor(x) for x in train_data])
# print(train_data[10].shape)