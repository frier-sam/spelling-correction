
# importing nltk and words only for the corpus of words 
# which i'll be using to train the model
# you can choose to expel these imports if you have a database of 
# words that can be trained

import nltk
import numpy as np
from nltk.corpus import words
from nltk.corpus import wordnet as wn
import string   



# following fuction will be caluculating the inverse document 
# frequency of each letter by using the counter which is just a 
# count of each letter in no. of words

def idfr(freq,counter):
    frequ = []
    for item in freq:
        frequ.append(np.array(item)*np.array(list(counter.values())))
    return frequ





# this fuction will be caluculating the term frequency of each letter
# taking a word as input and also updates the counter for idf fuction

def termfrequency(text,idf,counter):
    a = dict.fromkeys(string.ascii_lowercase, 0) 
    for l in list(text.lower()): 
        try:
            a[l] += 1    
        except:
            None
    for i in set(list(text.lower())):
        try:
            counter[i] +=1    
        except:
            None
    wo = len(list(text))
    for key in a.keys():
        a[key] = a[key]/wo
    return list(a.values())
 
 
 
    
# this fuction is used for calculation term frequency of predictioon word 
# as the before fuction updates counter and i don't wanna update counter
# with prediction word

def termfreq(text):
    a = dict.fromkeys(string.ascii_lowercase, 0) 
    for l in list(text.lower()): 
        try:
            a[l] += 1    
        except:
            None      
    wo = len(list(text))
    for key in a.keys():
        a[key] = a[key]/wo
    return list(a.values())




# this fuction handels the termfrequency iteration over all words in a corpus
# provided as input and creates the dictionary of words that 
# can be lookedup later
# This is the only uctions you will be using to training data
# as it handels all the fuctions internally except termfreq

def tfidf(li,idf=True):
    allwords = li
    counter = dict.fromkeys(string.ascii_lowercase, 0)
#     r = dict.fromkeys(string.ascii_lowercase, 0)
    freq = []
    word_dict = {}
    for idw,word in enumerate(allwords):    
        word_dict[idw] = word
        freq.append(termfrequency(word,idf,counter))
#     print(counter)
    if bool(idf):
        tot = len(words.words())
        for key in counter.keys():
            counter[key] = tot/counter[key]
        freq = idfr(freq,counter)
    return freq,word_dict,counter


#this fuction takes a word as input along with no. predictions needed and
#counter and word_dict that we will get back when training the model(see below) 
#this fuction calculates k nearest neighbours in corpus and provides 
#nearest n results as outpus searching the words in word_dictionary

def correct(text,n,counter,w_d):
    pword = np.array(termfreq(text)).T
    pp = np.array(pword)*np.array(list(counter.values()))
    res = {}
    for idr,row in enumerate(freq):
        res[np.sqrt(np.sum((np.array(pp)-np.array(row))**2))] = idr  
    final = []
    for k in list(sorted(res.keys()))[:n]:
        final.append(w_d[res[k]])
    return final
    

########### USAGE ##############
    
#### TRAINING ######   

#tarining the model with corpus from nltk, you can change the corpus to
#your own data.
#out-put is tf-idf of allwords, word dictionary for lookup,counter for calculation tfidf of prediction word
#you can see we are getting frequency(tf-idf),word_dictionary for lookup,counter for caluculating tf-idf of prediction word

corpus = list(set(words.words()))
freq,w_d,counter = tfidf(corpus,idf=True) 




##### PREDICTION #####

#once the model is trained you can get the prediction by calling the below fuction
#inputs (prediction word,no. of predictions, counter for training,word dictionary from training)
#output will be a list of predictions


correct('fallng',5,counter,w_d)
##OUTPUT##
['falling', 'fangle', 'Fingal', 'flagon', 'felling']

#### tfidf and k nearest neighbours algorithems are written from scratch
#### using numpy , so this is computationally heavy, you can use 
#### sklearn or any relavent libraries for a performance improvement
