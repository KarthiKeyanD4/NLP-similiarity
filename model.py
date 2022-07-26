#!/usr/bin/env python
# coding: utf-8

# In[7]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle


def compare(X,Y):
    # tokenization
    X_list = word_tokenize(X) 
    Y_list = word_tokenize(Y)
  
    # sw contains the list of stopwords
    sw = stopwords.words('english') 
    l1 =[];l2 =[]
  
    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw} 
    Y_set = {w for w in Y_list if not w in sw}
  
    # form a set containing keywords of both strings 
    rvector = X_set.union(Y_set) 
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)
    c = 0
  
    # cosine formula 
    for i in range(len(rvector)):
        c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return(cosine)


# In[8]:


X = input("Enter first string: ").lower()
Y = input("Enter second string: ").lower()


# In[9]:


pickle.dump(compare, open('LRClassifier.pkl', 'wb'))
# load the model from disk
loaded_model = pickle.load(open('LRClassifier.pkl', 'rb'))
result = loaded_model(X,Y)
print(result)


# In[ ]:




