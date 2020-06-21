# ------------------------------------------------------------------------------------------------------------------
# Import the required libraries
# ******************************************************************************************************************

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import os
import plotly.graph_objs as go
import plotly
from libraries.analytic import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from libraries.analytic import *
import json
# ------------------------------------------------------------------------------------------------------------------
# 1. Remove the news rows for which the full news were not taken due to scraping issues
# 2. Include the inputs taken from the front end screen
# 3. Sort the Entities with respect to the descending order of the occurences
# 4. Create a final list of entities which needs to be checked
# 5. Iterate over each of the entities to search for the sentiment which the news have with respect to each entity
# 6. Save the Entities and their scores seperately
# 7. Process the data in order to get graph into a desirable format and save on the disk
# ******************************* Preprocessing input words ********************************************************

def processsenti(data):
    #data=pd.read_excel('./Data/Newsfeed1.xlsx')
    list_data=[]
    data = data[data['Cleaned News'].notna()]
    list_data = data['Cleaned News'].tolist()
    score_list=[]
    score_list2=[]
    analyser = SentimentIntensityAnalyzer()

    result = []
    resultant_list = []
    
    # Include the inputs taken from the front end screen
    with open('./libraries/files/input_dict.json', 'r') as f:
        inputs = json.loads(f.read())
        
     
    # Extract required entities from the news    
    names = [inputs['inp_Org']]    
    for i in data['Cleaned News']:
        if inputs['inp_Org'].lower() in i.lower():
            try:
                result = result + derive_entities(i)['ORG']
            except:
                pass
            
    # Sort the Entities with respect to the descending order of the occurences
    for i in zip(pd.Series(result).value_counts().index, pd.Series(result).value_counts().values):
        resultant_list.append(i)
        
    #resultant_list_lower = [x.lower() for x in resultant_list]     
    # Create a final list of entities which needs to be checked
    names = names + [i[0] for i in sorted(resultant_list, key = lambda x: x[1], reverse = True)]
	
    names= set(names)
    # calcluate score for given name. Search entire corpus and identify scores
    def scores_valuation(item):
        score_list=[]
        for j in list_data :
            sentence = j
            try:
            
                if item.lower() in sentence.lower():
                    score = analyser.polarity_scores(sentence)  
                    score_list.append(score['compound'])
            except:
                continue

        avg_score=np.average(np.array(score_list))
        return avg_score	
        
    # Generated conslidated scores.
    for sj in names:
        result = scores_valuation(sj)
        score_list2.append([sj,result])
     
     
    color = list(np.random.rand(10))
    
    # Genereate graph	
    x_list =[]
    y_list =[]
    for i in range(0,len(score_list2)):
        x_list.append(score_list2[i][0])
        y_list.append(score_list2[i][1])
    data = [go.Bar(
                x=x_list,
                y=y_list,
                marker=dict(
        color=color) 
            )]
    layout = go.Layout(
            title='Sentiment score for Insurance Carriers under Watch List',
            barmode='stack',
            showlegend=False
            )

    fig = go.Figure(data=data)
    fig['layout'].update(layout)
    try:
        os.remove('./static/plots/sentiment2.html')
    except:
        pass
    plotly.offline.plot(fig, filename = './static/plots/sentiment2.html', auto_open=False)


