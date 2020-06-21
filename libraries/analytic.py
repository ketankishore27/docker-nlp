import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import json
import spacy
import textacy
from collections import OrderedDict, Counter
from spacy import displacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import numpy as np
#import gensim
import matplotlib.pyplot as plt
nlp = spacy.load('en_core_web_sm')
 
import en_core_web_sm

def derive_entities(sentence):

	# Wrapping a try- except block to handle the exception
	try:    
		
		# Loading the sentence
		doc = nlp(sentence)
		
		# Initializing an empty dictionary to store entities
		entities = {}
		
		# Iterating over the entities
		for i in set(doc.ents):
			
			if i.label_ in entities.keys():
				entities[i.label_].append(i.text)
			else:
				entities[i.label_] = [i.text]
			
		# Displaying the sentence according to the recognized entities
		displacy.render(doc, style = 'ent', jupyter = True)
		print('-'*50)

		return entities
		
	except Exception as e:
		
		print(str(e))
		return None
