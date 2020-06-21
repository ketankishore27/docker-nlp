FROM continuumio/anaconda3
WORKDIR /usr/app/Newsfeed

RUN apt update -y \
	&& apt install build-essential -y \
	&& apt-get install manpages-dev -y \
	&& pip install spacy==2.2.4 \
	&& pip install textacy==0.10.0 \ 
	&& python -m spacy download en_core_web_sm \
	&& pip install PyMsgBox \ 
	&& pip install networkx \
	&& pip install sklearn \
	&& pip install plotly \
	&& python -m nltk.downloader stopwords \
	&& pip install gensim \
	&& pip install wordcloud \
	&& pip install vaderSentiment 
#	&& rm -rf /var/cache/apt/*
COPY . .

EXPOSE 5003

CMD ["python", "Newsfeed_main_function.py"]
