#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
 
app = FastAPI()
 
class inp(BaseModel):
    X : str
    Y : str
        
@app.post('/predict')
async def predict_species(inp: inp):
    data = inp.dict()
    loaded_model = pickle.load(open('LRClassifier.pkl', 'rb'))
    data_in = [[data['X'], data['Y']]]
    compare = loaded_model.compare(data_in)
     
    return {
        'compare': compare
    }


# In[ ]:




