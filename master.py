import pandas as pd
import re

# Reading the text 
text = pd.read_table('text.txt', header=None)[0].tolist()[0]

# Removing punctuations
punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_“~'''
for x in text.lower(): 
    if x in punctuations: 
        text = text.replace(x, "")

# Cleaning the whitespaces
text = re.sub(r'\s+', ' ', text).strip()

# Setting every word to lower
text = text.lower()

# Converting all our text to a list 
text = text.split(' ')

# Getting all the unique words from our text 
words_unique = set(text)

