import warnings
warnings.filterwarnings("ignore")
from bs4 import BeautifulSoup
import requests
import time
import datetime

class content():

	def __init__(self, keys, region, number_of_news, days):

		
		self.keys  = keys
		self.region = region
		self.number_of_news = number_of_news
		self.days = days

		
	def scrap_headline_window(self):
    		
		# Putting it into try-catch to handle any error    
		try:
			data = []
			# Looping through the pages
			for page in range(0, self.number_of_news, 10):
				#print('Scraping_headline_window')

				# Defining a baseline URL 
				url= 'https://www.google.com/search?q={}+impact+on+%22{}%22&rlz=1C1GGRV_enIN790IN790&tbs=qdr:d{}&tbm=nws&ei=FlJhXLiKNNDikgXCkZTwDA&start={}&sa=N&ved=0ahUKEwi47dT2v7PgAhVQsaQKHcIIBc4Q8tMDCFE&biw=1440&bih=758&dpr=1'.format(self.keys, self.region, self.days, page); print(url)

				# Hitting the URL and Parsing it
				source = requests.get(url, verify = False)
				time.sleep(5)      # Reducing overload condition on the server else the IP might get blocked

				# Checking the Status code for successful request
				if source.status_code == 200:
					print('.', end = '')
					source = BeautifulSoup(source.text, 'lxml')
						
					# Scraping the news and the respective links
					for elements in source.find_all('div', class_ = 'kCrYT'):
						#print('1')
						#print(elements)
						heading = elements.find().text
						#print('2')
						if(elements.find('a')) is not None:
						    link = '.' + elements.find('a')['href'].split('=')[1][:-3]
						    #print('.'+link, end = '')
						    retrieved_google_info =  self.content(link)
						    #print('\nretrieved_google_info=' + retrieved_google_info, end = '')
						    #news_source, event_time = self.retrieve_time(elements)
						    #print('\n**', end = '')encode('utf8')
						    data.append([heading, retrieved_google_info.encode('utf-8'), 'NONE', 'NA', link])
						    #print('\ndata  ', end = '')
				else:
					print()
					print('Request failed with status code', source.status_code())
						
		except Exception as e:
			
			print('\n Error - ' + str(e))
			pass
			return ""
			
		return data
				

			
	def content(self, link):
    		
		try:
        		
			# Requesting the Scraped Link
			time.sleep(2)     # Reducing number of pings to avoid ip blocks though not neccessary at this stage
			source = requests.get(link[1:], verify = False)
			context = ' '

			# Check if the request is successful
			if source.status_code == 200:
				print('.', end = ' ')

				# Parsing the link text
				source = BeautifulSoup(source.text, 'lxml')
	
				# Finding the Paragraph text using 'p' tag
				for data in source.find_all('p'):
		
					# Implementing a bit of cleaning text at initial stage
					if len(data) < 2:
						data = data.text.replace('  ', '')
						context = context + ' ' + data.rstrip().lstrip().rstrip('\n').lstrip('\n').replace('\n', ' ')
					elif data.find('a'):
						data = data.text.replace('  ', '')
						context = context + ' ' + data.rstrip().lstrip().rstrip('\n').lstrip('\n').replace('\n', ' ')
			else:
				print()
				print('Requested site for content did not respond. Status Code is ', source.status_code)

		except Exception as e:
			print(str(e))
			context = ' '

	
		return context
			
			
	def retrieve_time(self, source):
    
		for element in source.find_all('div', class_ = 'slp'):

			news_source, time = element.text.split('-')[0].rstrip().lstrip(), element.text.split('-')[1].rstrip().lstrip()

			if 'hours' in time or 'minutes' in time or 'seconds' in time or 'hour' in time or 'minute' in time or 'second' in time:
				return news_source, str(datetime.datetime.utcnow().date())
			elif 'days' in time or 'day' in time:
				return news_source, str((datetime.datetime.utcnow() - datetime.timedelta(days = int(time.split()[0]) + 1)).date())
			else:
				return news_source, time