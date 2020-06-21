# ------------------------------------------------------------------------------------------------------------------
# Import the required libraries
# ******************************************************************************************************************

import re
import numpy as np
import os
import sys
import random
import pandas as pd
import matplotlib.pyplot as plt 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ------------------------------------------------------------------------------------------------------------------
# 1. Split into tokens by white space
# 2. Remove punctuation from each token
# 3. Remove remaining tokens that are not alphabetic
# 4. Filter out stop words
# 5. Convert data into lower case
# 6. Filter out short tokens
# ******************************* Preprocessing input words ********************************************************

def Datapreprocess(data):

    cat_data = data
    cat_data.head(5)
    cat_data.columns
    len(cat_data)
    Full_News=cat_data['Scraped Raw News']
    Full_News = [incom for incom in Full_News if str(incom) != 'nan']
    type(Full_News)
    
    with open('./libraries/files/stopwords.txt') as ads_word:

        stop_words = []
    
        for i in ads_word.readlines():
            if len(i) > 1:
                stop_words.append(i.strip())
    def clean_doc(doc):
    
    #Split into tokens by white space
        tokens = str(doc).split()
        
	# remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)        
        tokens = [w.translate(table) for w in tokens]
        
	# remove remaining tokens that are not alphabetic
        tokens = [word.lower() for word in tokens if word.isalpha()]
        
	# filter out stop words
        stop_words = set(stopwords.words('english')+ ['share', 'email', 'facebook', 'messenger', 'twitter', 'pinterest', 
                                                   'whatsapp', 'linkedin'])

	#Convert data into lower  
        tokens = [w for w in tokens if not w in stop_words]
        
	# filter out short tokens. Convert data to lower
        tokens = [word.lower() for word in tokens if len(word) > 2]
		
        return tokens
        
    # Applying the above function to all the news documents
    Full_News_tokens = []
    for doc in Full_News:
        tokens = clean_doc(doc)
        Full_News_tokens.append(tokens)
    

    return(Full_News_tokens)
