# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------------------------------------------
# Import the required libraries
# ******************************************************************************************************************

from flask import Flask, redirect, url_for, request,render_template
import pandas as pd
import numpy as np
import datetime
import pymsgbox
import io
import seaborn as sns
import os
from libraries.analytic import *
from libraries.content_scrap import content
from libraries.clean_text import *
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from flask import send_file 
import Newsfeed_Preprocessing
import pandas as pd
import time
import string
from flask import jsonify
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
import json
import warnings
warnings.filterwarnings('ignore')
import Newsfeed_Sentiment_analysis
from Newsfeed_scraping import fetch_webscrape_data
from Newsfeed_generate_networkgraph import newsfeed_network_graph
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import gensim
import rasterio
import os
import reverse_geocoder as rg
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# ------------------------------------------------------------------------------------------------------------------
# 1. Initialize values which would be used to store the inputs from front endswith
# 2. Intialize the values which would be used to pass values from one function to other
# **************************** Initialize input variables **********************************************************

f_input_data= ''
fh_summary=''
fh_summary_2=''
inp_Keyword=''
inp_Country = ''
inp_AmtofData =''
inp_Duration = ''
inp_Stdate = ''
inp_st1date = ''
inp_st2date ='' 

# ------------------------------------------------------------------------------------------------------------------
# Initialize the Flask Application          
# **************************** Initialize Flask Application ********************************************************

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

# ------------------------------------------------------------------------------------------------------------------
# 1. Read the news extracted
# 2. Load the pre trained Word2vec model 
# 3. Includes additional stopwords which needs to be eliminated
# 4. Perform the cleaning and text preprocessing parts
# 5. Measure the similarity score on each word with respect to each of the other words
# 6. Get the corresponding to word vectors and assign similar worrd vectors to similar words with respect to meaning
# 7. Apply a TSNE transform with PCA method to aggregate the results into one columns
# 8. Scale the word magnitudes formed after transformation to fall into a specified range
# 9. Plot the same on a graph and save as an HTML file
# **************************** Initialize Flask Application ********************************************************

def process_word2vec():
    
    print("inside w2v")
    full_news = input_data['Cleaned News']
    
# ********************** Catching exception for improper load of word2vec model ***********************************
    
    try:
        
        model = gensim.models.KeyedVectors.load_word2vec_format('./Data/GoogleNews-vectors-negative300.bin', binary=True, limit = 100000) 
        
    except Exception as e:
        
        print('Error Loading word2vec model')
        print(str(e))
        pymsgbox.alert('Error Loading the Word2vec model', 'Please check if the Word2Vec model is at specified location')
        
# -----------------------------------------------------------------------------------------------------------------
        
    full_news_token = []
    
# **********************  Catching exception for improper load of files ********************************************
    
    try:
        
        with open('./libraries/files/stopwords.txt') as ads_word:
    
            stop_words = []
        
            for i in ads_word.readlines():
                if len(i) > 1:
                    stop_words.append(i.strip())
                    
    except:
        
        print('Improper Load of files')
        pymsgbox.alert('Improper Load of files', 'Please check the location of stopword files')
                    
# ------------------------------------------------------------------------------------------------------------------
                    
# ******************************* Preprocessing input words ********************************************************
    try:
        
        for doc in full_news:
            tokens = str(doc).split()
            table = str.maketrans('', '', string.punctuation)
            tokens = [w.translate(table) for w in tokens]
            tokens = [word.lower() for word in tokens if word.isalpha()]
        
            stop_words = set(stopwords.words('english') + ['share', 'email', 'facebook', 'messenger', 'twitter', 'pinterest', 
                                                           'whatsapp', 'linkedin'])
            tokens = [w for w in tokens if not w in stop_words]
            tokens = [word.lower() for word in tokens if len(word) > 2]
            full_news_token += tokens
        
        final_token_list = set(full_news_token)
    
    except:
        
        print('Error in pre processing inputs')
        pymsgbox.alert('Please check', 'Please check preprocessing part or the content of news')

# ------------------------------------------------------------------------------------------------------------------
        
# ******************************** Similarity Measures part ********************************************************
        
    try:
    
        similarity_measure = {}
        for parent_word in final_token_list:
            for child_word in final_token_list:
                try:
                    score = model.similarity(parent_word, child_word)
                    if score > 0.5 and parent_word!=child_word:
                        #print(score)
                        if parent_word not in similarity_measure.keys():
                            similarity_measure[parent_word] = np.array([0] * 300)
                        
                        similarity_measure[parent_word] = similarity_measure[parent_word] + model.get_vector(child_word)
                except:
                    pass
        embedding_frame = pd.DataFrame(similarity_measure.values())
        embedding_frame = pd.DataFrame.from_dict(similarity_measure).T
        tsne_model = TSNE(perplexity=20, n_components=1, init="pca", n_iter=5000, random_state=23)
        Y_frame = tsne_model.fit_transform(embedding_frame)
        
    
        scaler = MinMaxScaler(feature_range = (0, 5))
        embedding_frame['Resultant'] = np.array(scaler.fit_transform(Y_frame))
        
        indexes = np.linspace(0, embedding_frame.shape[0] - 1, 100).astype('int')
        temp_data = embedding_frame.sort_values(by = 'Resultant', ascending = False)['Resultant'][indexes].sample(80)
     
        
        fig = go.Figure(data=[go.Scatter(
                x=temp_data.index, y=temp_data,
                mode='markers',
                marker=dict(
            color=temp_data.values.tolist(),
            showscale=True
            ),
            marker_size=(temp_data.values * 5).tolist()), 
                ])
    
        fig.update_layout(
            title={
                'text': "Semantic Word Representation being used Frequently", 'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}, 
            xaxis_title="Semantic Words into base Representation",
            yaxis_title="Magnitude",
        
        )
        
        try:
            os.remove('./static/plots/Word2vec.html')
        except:
            pass
            
        plotly.offline.plot(fig, filename = './static/plots/Word2vec.html', auto_open=False)
        print("word2vec finished")
        return None
    
    except:
        
        print('Error in the Word2vec processing')
        print(str(e))
        pymsgbox.alert('Error in creating Word2Vec graph', 'Please check the content of the scraped News or dependancies in the environment')
        
# ------------------------------------------------------------------------------------------------------------
# Route the Application to the index page
# ******************************** Index  Page Routing********************************************************
@app.route('/')
def index():
    return render_template('index.html')
    
# -------------------------------------------------------------------------------------------------------------
# Route the application to the page where we give inputs
# ******************************** Explore Page Routing********************************************************
	
@app.route('/input',methods = ['GET','POST'])
def explore():
     return render_template('input.html')
     
# -------------------------------------------------------------------------------------------------------------
# 1. Take the inputs and send the inputs to all the underlying functionalities to preprocessing
# 2. Do all the pre processing before we going to the insights page so that insights page dosent take long to generate
# 3. Save the inputs for further references
# ******************************** Fetch Page Routing**********************************************************
	 
@app.route('/fetching',methods = ['GET','POST'])
def fetching():
    
     input_dict={}
     input_dict['inp_Keyword'] = request.form['Keyword']
     input_dict['inp_Country'] = request.form['Country']
     input_dict['input_amt_data'] = request.form['Duration']
	 
     try:
         input_dict['inp_Days'] = request.form['Days']
		 
     except:
         pass
     try:
         input_dict['inp_Org'] = request.form['Org']
		 
     except:
         pass
		 
     try:
         input_dict['inp_st1date'] = request.form['st1date']
         input_dict['inp_st2date'] = request.form['st2date']
     except:
         pass

     with open('./libraries/files/input_dict.json', 'w') as f:
         f.write(json.dumps(input_dict))
         
     scrapped_file = fetch_webscrape_data(input_dict)
     print("scrapped_file",scrapped_file.head())
     #Build the network graph for the scraped data
     try: 
         newsfeed_network_graph(scrapped_file)
         
     except:
         print('Error in Generating the filtered Network Graph')
         pymsgbox.alert('Network Graph Error', 'Error in Generating the filtered network graph')
         
        
     print("network_graph_finished")
     print("func word2vec")
     process_word2vec()
     return render_template('fetching.html',fh_input_data=input_dict)
     
# -------------------------------------------------------------------------------------------------------------
# Take a copy of data for further processing
# ******************************** Backing up input Data ******************************************************
 
try:  
    input_data=pd.read_excel('./Data/Scraped_raw_stage1.xlsx')
except:
    print('Please proceed to scraping first')
    pass

@app.route('/graphs',methods = ['GET','POST'])
def graphs():
    return render_template('graphs.html')

# -------------------------------------------------------------------------------------------------------------
# 1. Send the data to Sentiment Analytics module to preprocess data
# 2. Extract the Named Entities which we would like to include the graph
# 3. Iterate over each of the entities to search for the sentiment which the news have with respect to each entity
# 4. Save the Entities and their scores seperately
# 5. Process the data in order to get graph into a desirable format and save on the disk
# ******************************** Sentiment Analysis Page Routing ********************************************

@app.route('/sentiment')
def sentiment():

    data = input_data

    Newsfeed_Sentiment_analysis.processsenti(input_data)

    return render_template('sentiment.html')
    
# -------------------------------------------------------------------------------------------------------------
# The processing for the same is done before the application comes to fetching page results 
# Therefore we just need to load the file created during the processing
# ******************************** Network Graph Page Routing *************************************************

@app.route('/networkgraph')
def networkgraph():
    
    with open('./libraries/files/input_dict.json', 'r') as f:
        inputs = json.loads(f.read())
    return render_template(inputs['network_timestamp'].split('/')[2])

# -------------------------------------------------------------------------------------------------------------
# The processing for the same is done before the application comes to fetching page results 
# Therefore we just need to load the file created during the processing    
# ******************************** Cluster Analysis Page Routing **********************************************
	
@app.route('/cluster1')
def cluster1():
    return send_file(img, mimetype='image/png')
    
# -------------------------------------------------------------------------------------------------------------
# 1. Preprocess the data to convert it into a sparse vector. 
# 2. Apply methods to remove the frequent words still left
# 3. Get the top feature from the sparse matrix
# 4. Apply a transformation to convert the sparse matrix into a column of two vectors
# 5. Apply a clusterig algorithm to make group for the words filtered out
# 6. Represent the same in the graph and save it for further loads
# ******************************** Word CLuster Page Routing **************************************************
	
@app.route('/wordcluster')
def wordcluster():

    Cleaned_data=[]
    Cleaned_data=Newsfeed_Preprocessing.Datapreprocess(input_data)

    Full_News_formatted = []
    for i in Cleaned_data:
        tokenized_content_formatted = ' '.join(i)
        Full_News_formatted.append(tokenized_content_formatted)  

    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tf_idf = tfidf_vect.fit_transform(Full_News_formatted)
    
    tf_idf_norm = normalize(tf_idf)
    tf_idf_array = tf_idf_norm.toarray()

    n_clusters = 3
    sklearn_pca = PCA(n_components = 2)
    Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)
    kmeans = KMeans(n_clusters= n_clusters, max_iter=600, algorithm = 'auto')
    kmeans.fit(Y_sklearn)
    prediction = kmeans.predict(Y_sklearn)
    
    
    def get_top_features_cluster(tf_idf_array, prediction, n_feats):
        labels = np.unique(prediction)
        dfs = []
        for label in labels:
            id_temp = np.where(prediction==label) # indices for each cluster
            x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
            sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
            features = tfidf_vect.get_feature_names()
            best_features = [(features[i], x_means[i]) for i in sorted_means]
            df = pd.DataFrame(best_features, columns = ['features', 'score'])
            dfs.append(df)
        return dfs
    dfs = get_top_features_cluster(tf_idf_array, prediction, 20)
	
    ###Code for plotly word cluster graph
    x_feat0=dfs[0][:15]['features']
    y_feat0=dfs[0][:15]['score']
    x_feat1=dfs[1][:15]['features']
    y_feat1=dfs[1][:15]['score']
    x_feat2=dfs[2][:15]['features']
    y_feat2=dfs[2][:15]['score']
    
    trace0 = go.Bar(
    x=y_feat0,
    y=x_feat0,
    textposition = 'auto',
    orientation = 'h',
    name='Cluster1'
    )
    trace1 = go.Bar(
            x=y_feat1,
            y=x_feat1,
            textposition = 'auto',
            orientation = 'h',
            name='Cluster2'
            )
    trace2 = go.Bar(
            x=y_feat2,
            y=x_feat2,
            textposition = 'auto',
            orientation = 'h',
            name='Cluster3'
            )

    layout = go.Layout(
            title='Clusters analysis',
            )
    # Creating two subplots
    fig = tools.make_subplots(rows=1, cols=3, specs=[[{}, {},{}]], shared_xaxes=True,
                              shared_yaxes=False, vertical_spacing=0.001)
    
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace2, 1, 3)
    
    fig['layout'].update(layout)

    try:
        os.remove('./static/plots/Clusters3.html')
    except:
        pass
    
    plotly.offline.plot(fig, filename = './static/plots/Clusters.html', auto_open=False)
    
            # Start with one review:
            # Start with one review:
    text = dfs[0][:50].features.tolist()
    test1=' '.join(text)
    text2 = dfs[1][:50].features.tolist()
    test2=' '.join(text2)
    text3 = dfs[2][:50].features.tolist()
    test3=" ".join(text3)
    stop_words = set(stopwords.words('english'))
    # Generate a word cloud image
    print("test1",test1)
    print("test2",test2)
    print("test3",test3)
    wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(test1)
    
    # Display the generated image:
    # the matplotlib way:
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    try:
        os.remove('./static/plots/cluster1.png')
    except:
        pass
    wordcloud.to_file("./static/plots/cluster1.png")

    wordcloud2 = WordCloud(stopwords=stop_words, background_color="white").generate(test2)
    plt.figure()
    plt.imshow(wordcloud2, interpolation="bilinear")
    plt.axis("off")
    try:
        os.remove('./static/plots/cluster2.png')
    except:
        pass
    wordcloud2.to_file("./static/plots/cluster2.png")
    
    wordcloud3 = WordCloud(stopwords=stop_words, background_color="white").generate(test3)
    plt.figure()
    plt.imshow(wordcloud3, interpolation="bilinear")
    plt.axis("off")
    try:
        os.remove('./static/plots/cluster3.png')
    except:
        pass
    wordcloud3.to_file("./static/plots/cluster3.png")
    plt.tight_layout()

    return render_template('wordcluster.html')
    
    
# -------------------------------------------------------------------------------------------------------------
# Based on Static data, Needs more work to be done on the same  
# ******************************** COmpany Insight Page Routing ***********************************************

@app.route('/Insightscomp')
def Insightscomp():
    return render_template('Insightscomp.html')
    
# -------------------------------------------------------------------------------------------------------------
# Based on the image, Classify the image as one of the Catastrophe 
# -------------------------------------------------------------------------------------------------------------
    
@app.route('/upload')
def upload():  
    return render_template('upload.html')
    
def location_detect():

    os.system('gdalinfo -json ./Data/sample_data/harvey_tmo_2017237_geo.tif > ./Data/sample_location_info/info.txt')
    
    with open('./Data/sample_location_info/info.txt') as f:
      data = json.load(f)
      
    print(data)
    
    long, lat = data['cornerCoordinates']['upperLeft']
    
    print(long, lat)
    
    result_geo = rg.search([lat, long])
    result_dict = {}
    for result in result_geo[0].keys():
        result_dict[result] = result_geo[0][result]
    print(result_dict)
    
    return result_dict
    
@app.route('/prediction', methods=['POST'])
def prediction():

    
    print('using Tensorflow version: {}'.format(tf.__version__))

    model = load_model('./Data/satimageweatherpred.h5')

    img = load_img('./Data/sample_data/harvey_tmo_2017237_geo.tif', target_size=(256, 256))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)/255
    pred = model.predict(x)
    print(pred)
    pred_labels = {'0': 'Hurricane', '1': 'Flood', '2': 'Tropical Storm'}
    
    location_dict = {}
    location_dict['value'] = pred_labels[str(np.argmax(pred))]
    location_dict.update(location_detect())
    
    return jsonify(location_dict)


    

#==============================================================================
# Run the application on specified port
#==============================================================================

if __name__ == '__main__':
    #app.run(host='10.109.32.238', port=5003)   # Configured to run on Port 5003
    app.run(host='0.0.0.0', port=5003, debug = False)   # Configured to run on Port 5003
