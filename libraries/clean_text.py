import spacy
import textacy
import en_core_web_sm

nlp = spacy.load('en_core_web_sm')
with open('./libraries/files/stopwords.txt') as ads_word:

    stopword = []
    for i in ads_word.readlines():
        if len(i) > 1:
            stopword.append(i.strip())

def cleaning_text(content):
    
	filtered_message = ''  # Resetting the filtered message for every token

	# Implementing the doc for using it to split using spacy

	try:          # Wrapping this because some of the content field is empty
		
		doc = nlp(content)
		for word in doc.sents:
			
			# Wrapping in try-catch to ensure exception handling
			try:
				
				# Since the word is of span class, thus type casting it to be a string
				word = str(word)
				spam = 0      # Setting the spam flag, whenever it is set to 1, the sentence will be discarded
					
				for ad_word in stopword:     # Checking for stopword
					if (ad_word in word.lower()):
						spam = 1
						break
						
				if len(word.split())< 5:  # Checking for length of the sentence
					spam = 1
				
				if spam == 0:     # If flag does not changes, then adding it to the filtered message
					filtered_message = filtered_message + word + ' '
					
			except Exception as e:
				
				print(str(e))
				pass

		filtered_message = filtered_message.encode('ascii', 'ignore').decode('unicode_escape')
		#filtered_message =  textacy.preprocess_text(filtered_message, fix_unicode=True, lowercase=False, transliterate=True, no_urls=True, no_emails=True, no_phone_numbers=True, no_currency_symbols=True, no_punct=True, no_contractions=True, no_accents=True)
		return filtered_message.replace('\n', ' ').replace('\t', ' ')
    
	except Exception as e:          # Returning space when the content is empty 
		
		print('Exception Occured')
		print(str(e))
		#print(content)
		return ' '
