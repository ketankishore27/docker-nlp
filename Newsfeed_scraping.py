#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------------------------
# Import the required libraries
# ******************************************************************************************************************

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from nltk.tokenize import sent_tokenize
import spacy
import itertools
import networkx as nx
import pymsgbox
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from libraries.analytic import *
from libraries.content_scrap import content
from libraries.clean_text import *

# ------------------------------------------------------------------------------------------------------------------
# 1. This module calls the web scrap module and passes the input parameters
# 2. The extracted results are checked for any duplicates values if any
# 3. The modules then calls a text cleaning module which does the basic text cleaning like removing punctuation or
#    removing utf-8 characters scraped out
# 4. The data is then added to the existing data structure
# 5. Entities are derived at this point and added to the data structure in dictionary format
# 6. The existing data structure is converted to dataframe and written on to disk as a backup
# ******************************************************************************************************************

def fetch_webscrape_data(key_input_dict):

    def web_scrape(keyword, org_name, region, amount_of_data,no_of_days):

# ******************************Scraping Google News ****************************************************
        try:
            web_data = content(keyword, org_name, amount_of_data, no_of_days)
            raw_text = web_data.scrap_headline_window()
        except:
            print("Please check your internet conection")
            pymsgbox.alert('Scraping Issue', 'Please check the Internet Connectivity. Else look for html handles on Google')
            
        print("raw_text",raw_text)
        rawdf = pd.DataFrame(raw_text, columns = ['Headlines', 'News', 'News Source', 'News Date', 'News Link'])
        rawdf.drop_duplicates(subset=['News', 'News Source', 'News Date','News Link'],inplace=True)
        #raw_text_org = raw_text
        raw_text = rawdf.values.tolist()
        return raw_text
    
# ---------------------------------------------------------------------------------------------------------
        
# ****************************** Calling Web Scrap function ************************************************
    
    try:
        keyword = key_input_dict['inp_Keyword']
        org_name = key_input_dict['inp_Org']
        region = key_input_dict['inp_Country']
        amount_of_data = key_input_dict['input_amt_data']
        print("amount_of_data",type(amount_of_data))
        print("amount_of_data",amount_of_data)
        no_of_days = key_input_dict['inp_Days']
        print("no_of_days",type(no_of_days))
        print("no_of_days",int(no_of_days))
        raw_text = web_scrape(keyword, org_name, region, int(amount_of_data),int(no_of_days))
    
    
        rawdf = pd.DataFrame(raw_text, columns = ['Headlines', 'News', 'News Source', 'News Date', 'News Link'])
        rawdf.drop_duplicates(subset=['News', 'News Source', 'News Date','News Link'],inplace=True)
        raw_text = rawdf.values.tolist()
    
        rawdf[rawdf['News'] != ''] 
        
    except:
        
        print('Exception in calling the Scrap function')
        pymsgbox.alert('Scraping Issue', 'Exception in calling the Scrap Function')

# ----------------------------------------------------------------------------------------------------------

# ****************************** Cleaning the Raw text scraped from internet ************************************************
    try:
        
        for i in raw_text:
            
            i.insert(2, cleaning_text(i[1].decode("utf-8")))
    
    except:
        
        print('Exception in cleaning the text')
        pymsgbox.alert('Cleaning Text Issue', 'Please check the Spacy and Textacy compatibility and debug the same')

# ----------------------------------------------------------------------------------------------------------


# ****************************** Derive Entities Function ************************************************
    try:
        
        for i in raw_text:
            entities = derive_entities(i[2])
            
            for j in ['PERSON', 'GPE', 'ORG', 'EVENT', 'CARDINAL', 'MONEY', 'FAC']:
                if j in entities.keys():
                    i.append(set(entities[j]))
                else:
                    i.append(" ")
                    
    except:
        
        print('Exception in deriving entities')
        pymsgbox.alert('NER Issue', 'Exception in extracting the Named Entity extraction')

# ----------------------------------------------------------------------------------------------------------

# ****************************** Writing data to disk Function ************************************************
    try:
        
        raw_scraped_data = pd.DataFrame(data = raw_text, columns = ['Headlines', 'Scraped Raw News', 'Cleaned News', 'News Source', 'News Date', 'News Link', 'Person', 'Location', 'Organization', 'Event', 'Cardinal', 'Money', 'Fac'])
        raw_scraped_data.to_excel("./Data/Scraped_raw_stage1.xlsx")
        
    except:
        print('Exception in writing the data to disk')
        pymsgbox.alert('Writing data to Disk Issue', 'Please check the number of columns and specfied. There can be a mismatch')
    
    return raw_scraped_data

# ----------------------------------------------------------------------------------------------------------

