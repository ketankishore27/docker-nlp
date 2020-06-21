
#####
#This module contains the code related to generation of Network graph 
#Steps involved are
#1. Extract the raw data from main module and derive ORG names and PERSON entities
#   from the data
#2. From the raw news, as a first step remove all these ORG names and PERSON related 
#   information
#3. Now perform tokenization for the data and remove the stop words from the data.
#4. Load the COvid related word2vec and consider only the words relavant to covid
#   using the function similarity_measure_metric
#

################################################################################
#Improting the libraries
##################
import warnings
warnings.filterwarnings("ignore")
from nltk.tokenize import sent_tokenize
import spacy
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from libraries.analytic import *
from libraries.content_scrap import content
from libraries.clean_text import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
import time
import re
import glob
import json
import os
import string
from gensim.models import Word2Vec
nlp = spacy.load('en_core_web_sm')
# ## All Functions
######################################################################################

#********************************Main function***************************************

def newsfeed_network_graph(scraped_df):
    print("inside network graph",scraped_df.head(1))

#****************************Derive entities **********************************
#  This derive entities wouldextract the named entities and and returns the list
#  of these entities
#******************************************************************************    
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
    		return entities
    		
    	except Exception as e:
    		print(str(e))
    		return None
    
#****************************clean_text_stopword **********************************
#  clean the text and remove stop words
#******************************************************************************
    def clean_text_stopword(sentence):
        tokens = str(sentence).lower().split()
        word = ' '.join([w for w in tokens if not w in stop_words])
        return word
    
    
#****************************Generate_html*************************************
#  Creates the necessary files for creation of the network graph
#******************************************************************************   
    def generate_html():
        
        try:
            for path in glob.glob('./templates/network_dynamic*.html'):
                os.remove(path)
        except:
            print('Delete Error')
            pass
        
        with open('./libraries/files/networkgraph_head.txt', 'r') as head:
            head_code = head.read()
            
        with open('./libraries/files/networkgraph_tail.txt', 'r') as tail:
            tail_code = tail.read()
            
        with open('./libraries/files/sample.json', 'r') as data:
            info = data.read()
            
        with open('./libraries/files/nodes_info.txt', 'r') as nodes:
            node_info = nodes.read()
            
        network_html = head_code + info + ', nodes:'  +  node_info + tail_code

        with open('./libraries/files/input_dict.json', 'r') as f:
            inputs = json.loads(f.read())
        with open('./libraries/files/input_dict.json', 'w') as f:
            inputs['network_timestamp'] = './templates/network_dynamic{}.html'.format(str(time.time()).split('.')[0])
            f.write(json.dumps(inputs))
            
        with open(inputs['network_timestamp'], 'w') as output:
            output.write(network_html)
            
        print('Dynamic Network Graph Generated')
        
        return None
    
#****************************similarity_measure_metric*************************************
#  This funciton would compare the news data corpus with word2vec model to extract
#  only relavant corpus for further processing        
#******************************************************************************   
   
    def similarity_measure_metric(input_list,score_cutoff,search_items,new_model):
        similarity_measure = {}
        #useful_tokens = 
        #searach_word_list = ['covid','corona','fever','virus','cough']
        searach_word_list = search_items
        for parent_word in input_list:
            for child_word in searach_word_list:
                try:
                    score = new_model.similarity(parent_word, child_word)
                    #print("score",score)
                    if score > score_cutoff and parent_word!=child_word:
                        if parent_word not in similarity_measure.keys():
                            similarity_measure[parent_word] = np.array([0] * 100)
                        similarity_measure[parent_word] = similarity_measure[parent_word] + new_model[parent_word]
                except:
                    pass
    
        embedding_frame_T = pd.DataFrame.from_dict(similarity_measure).T
        return(embedding_frame_T)
        
#****************************clean_doc*************************************
#  This funciton tokenize the data and performs cleaning       
#******************************************************************************   
  
    def clean_doc(doc):
        try:
         #Split into tokens by white space
            tokens = str(doc).split()
    	# remove punctuation from each token
            table = str.maketrans('', '', string.punctuation)
            tokens = [w.translate(table) for w in tokens]
    	# remove remaining tokens that are not alphabetic
            tokens = [word.lower() for word in tokens if word.isalpha()]
    	# filter out stop words
            stop_words = set(stopwords.words('english')+ ['share', 'email', 'facebook', 'messenger', 'twitter', 'pinterest', 'raja','trump','cases','likely','also','mahindra','please','provide',
                                  'anand' ,'whatsapp', 'linkedin','said','new','would','morgan','stanleys','stanley','according'] + orgs_stopwords_lower_new + names_stopwords_lower_new)
    #         stop_words = set(stopwords.words('english'))
    	#Convert data into lower  
        #tokens = [w for w in tokens if not w in stop_words]
            tokens = [w for w in tokens if not w.lower() in stop_words]
    	# filter out short tokens. Convert data to lower
            tokens = [word.lower() for word in tokens if len(word) > 2]
        except:
            pass
    		
        return tokens
#****************************Actaul process starts here*************************************
#  This funciton would compare the news data corpus with word2vec model to extract
#  only relavant corpus for further processing        
#******************************************************************************   

    data_scraped=scraped_df[scraped_df['Cleaned News'] != '']
    print("data_scraped",data_scraped.head(1))
##****Retrive all the Person info from the cleaned news and convert into list
    names=[]
    for i in data_scraped['Cleaned News']:
        try:
            entities = derive_entities(i)
            #print(entities)
            for j in ['PERSON']:
                if j in entities.keys():
                    names.append(entities[j])
        except:
            pass
    
    names_stopwords = []
    for sublist in names:
        for item in sublist:
            names_stopwords.append(item)
  
 ##****Retrive all the Orgnization info from the cleaned news and convert into list 
    orgs=[]
    for i in data_scraped['Cleaned News']:
        try:
            entities = derive_entities(i)
            #print(entities)
            for j in ['ORG']:
                if j in entities.keys():
                    orgs.append(entities[j])
        except:
            pass
   
    orgs_stopwords = []
    for sublist in orgs:
        for item in sublist:
            orgs_stopwords.append(item)
    
  
 ##*********convert the data into lower case  ********************************
    orgs_stopwords_lower = [x.lower() for x in orgs_stopwords] 
    names_stopwords_lower = [x.lower() for x in names_stopwords] 
    
 ##*********Remove the duplicates from above lists  ***************************     
    orgs_stopwords_lower_new = list(set(orgs_stopwords_lower))
    names_stopwords_lower_new = list(set(names_stopwords_lower))
    
    

####Word embeding code  
    Full_News=data_scraped['Cleaned News']
    try:
        Full_News =[(lambda x: re.sub('[^A-Za-z0-9.]+', ' ', x))(str(x).lower()) for x in Full_News]
    except Exception as e:
        print(e)
        print('getting exception0')
    
    Full_News_stop =[]
    for item in Full_News:
        for word in orgs_stopwords_lower_new:
            item = item.replace(word, ' ')
        Full_News_stop.append(item)

    Full_News_stop_final =[]
    for item in Full_News_stop:
        for word in names_stopwords_lower_new:
            item = item.replace(word, ' ')
        Full_News_stop_final.append(item)

    with open('./libraries/files/stopwords.txt') as ads_word:
    
        stop_words = []
    
        for i in ads_word.readlines():
            if len(i) > 1:
                stop_words.append(i.strip())

    #print("Full_News_stop_final", Full_News_stop_final[3])
##*****************Clean and tokenize the data*******************************    
    Full_News_tokens = []
    for doc in Full_News_stop_final:
        tokens = clean_doc(doc)
        Full_News_tokens.append(tokens)
#********************************Load trained word2vec model for covid ******************    
    new_model = Word2Vec.load('./libraries/files/model_word1.bin')
#******************************************************************
    #********* Comparing Similarity of the data using word embeddings 
    # ### Filter content based on search_items and Score cut off
#******************************************************************************   

    search_items= ['covid','corona','fever','virus','cough','loans','slowdown','recession','default','losses']
    #Score cutoff will be used to identify the similarity of words using wordembeddings in similarity_measure_metric
    score_cutoff = 0.25
    relavant_news_content = []
    for news_content in Full_News_tokens:
        try:
            filterd_news_content=similarity_measure_metric(news_content,score_cutoff,search_items,new_model)
            #print(filterd_news_content)
            similarity_rate =len(filterd_news_content)/len(news_content)*100
            #Similarity rate is the percentage of words that are present in filtered data from each news item
            # if the percnetage of words in filtered content is more than 10 percent then only consider that news or else ignore
            if similarity_rate > 10:
                relavant_news_content.append(news_content) 
            else:
                relavant_news_content.append('This content can be ignored')
        except:
            pass

     
     
    print("relavant_news_content",relavant_news_content)
    data_scraped['relavant_news_content']=relavant_news_content
    

####Generating network graph    
    
    data_scraped_filtered = data_scraped[data_scraped['relavant_news_content'] != 'This content can be ignored']
 
    data_scraped_filtered.fillna('', inplace=True)
    data_scraped_filtered.to_excel('./Data/data_scraped_filtered.xlsx')
    
    
    def concat_string(sample):
         return ' '.join(sample)
    
   
    data_scraped_filtered['relavant_news_content_concat'] = data_scraped_filtered['relavant_news_content'].apply(concat_string)
    

    raw_string = ''
    
    for i in data_scraped_filtered['relavant_news_content_concat']:
        raw_string = raw_string + str(i) + ''
        
#***********************Create the Network graph*******************************    
#*************The network Work graph  logic     *******************************
#******************************************************************************        
    combining = pd.DataFrame(columns = ['from', 'to', 'relation'])
    for news in data_scraped_filtered['Cleaned News'][:40]:
    #    if 'Morgan Stanley' in news:
            doc = nlp(str(news))
            combination = []
            for sent in doc.sents:
                organizations = []
                doc1 = nlp(str(sent))
                for en in doc1.ents:
                    if en.label_ in ['ORG', 'LOC', 'GPE', 'EVENT', 'PERSON', 'NORP', 'FAC', 'PRODUCT', 'PERCENT']:
                        organizations.append(en)    
                pos_noun = []
                for token in doc1:
                    if token.pos_ in ['NOUN', 'PROPN']:
                        #if token.text in words_considered:
                        pos_noun.append(token.text)            
                pos_noun = set(pos_noun)
                pos_verb = []
                for token in doc1:
                    if token.pos_ in ['VERB', 'ADV']:
                        #if token.text in words_considered:
                        pos_verb.append(token.text)            
                pos_verb = set(pos_verb)
                for org in set(organizations):
                    for i in doc1.ents:
                        if i.label_ in ['MONEY', 'CARDINAL']: # or i.label_ in 'PERCENT': # or i.label_ in 'CARDINAL':
                            #print(i)
                            combination.append([str(org).replace(',', ' '), (str(i) + ' ' + ' '.join(pos_noun)).replace(',', ' '), ' '.join(pos_verb)])
                test = pd.DataFrame(data = combination, columns = ['from', 'to', 'relation'])
                combining = pd.concat([combining, test])
    
    combining = combining[(combining['from'] != combining['to'])]
    combining.drop_duplicates(inplace = True)
    
    to_use_data = combining #.iloc[:int(combining.shape[0]//3)]
    
    try:
        os.remove('./libraries/files/sample.json')
    except:
        pass
    
    to_use_data.to_json('./libraries/files/sample.json', orient = 'records')
    
    scaler = MinMaxScaler(feature_range = (5, 9))
    values = scaler.fit_transform(to_use_data['from'].value_counts().values.reshape(-1, 1))
    custom_data = pd.Series(data = values.flatten(), index = to_use_data['from'].value_counts().index)
    
    resultant = []
    for ent in custom_data.iteritems():
        entities_graph = {}
        entities_graph['id'] = ent[0]
        entities_graph['marker'] = {}
        entities_graph['marker']['radius'] = ent[1]
        entities_graph['marker']['fillColor'] = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])][0]
        resultant.append(entities_graph)

    try:
        os.remove('./libraries/files/nodes_info.txt')
    except:
        pass
    
    with open('./libraries/files/nodes_info.txt', 'w') as nodes:
        json.dump(resultant, nodes)
    
    generate_html()
    
    #plt.figure(figsize = (20, 19))
    #g = nx.from_pandas_edgelist(to_use_data, source='from', target='to', edge_attr = 'relation')
    #nx.draw_kamada_kawai(g, with_labels = True, node_size = 15, font_size = 10, node_color = 'r', edge_color = 'y', font_color = 'b')
    #plt.show()
        
    return None

    
    
    
