import pandas as pd
import numpy as np
import string
import spacy
import nltk
import re
from nltk.corpus import stopwords
from tqdm import tqdm as tq
from sklearn.model_selection import train_test_split

## Load the resources
nltk.download('stopwords')
nltk.download('gutenberg')
nltk.download('punkt')


def book_pagination(book, book_name):
    """
    Partitioning(pagination) of the book to take each 100 word with the label/book_name for each partition.
    """
    tokenized_words=nltk.word_tokenize(book)
    offset = 0
    pages = []
    for i in range(1, min(int(np.floor(len(tokenized_words)/100.0)), 200)):
        limit = i*100
        pages.append({'book_name': book_name,
                      'partition': " ".join(tokenized_words[offset:limit])})
        offset = limit 
    return pages
    
    
def clean_text(text):
  ## 1. Lowercase the text
  text = text.lower() 

  ## 2. Remove Punctuations
  text = text.translate(str.maketrans('', '', string.punctuation)) 
  
  ## 3. Tokenize all the words
  words = nltk.word_tokenize(text)

  ## 4. Remove stopwords and word digits
  clean_text = " ".join([ w for w in words if w.isalnum() ])
  clean_text = clean_text.replace("\t", ' ')
  # clean_text = " ".join([ w for w in words if w.isalnum() and (w not in stop_words)  ])
  return clean_text
  
  
  
  
books = ['milton-paradise.txt', 'shakespeare-caesar.txt', 'melville-moby_dick.txt', 'chesterton-brown.txt', 'whitman-leaves.txt']
book_pages = []

for book_name in tq(books):
    book = nltk.corpus.gutenberg.raw(book_name)
    clean_book = clean_text(book) ## Regex to clean the text and tokenize it
    mydict=book_pagination(clean_book, book_name.split('.txt')[0])
    book_pages+=mydict

    #book_pages+=book_pagination(clean_book, book_name.split('.txt')[0])
    
    

books_df = pd.DataFrame(book_pages)

## Split data to train and test sets for training and testing ML models

msk = np.random.rand(len(books_df)) < 0.8

train = books_df[msk]
train.reset_index(inplace=True)
del train['index']

test = books_df[~msk]
test.reset_index(inplace=True)
del test['index']

train.to_csv('train_books_partitions.csv')
test.to_csv('test_books_partitions.csv')
books_df.to_csv('gutenberg_books_partitions.csv')
