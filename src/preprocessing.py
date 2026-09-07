import numpy as np
import pandas as pd

def add_housekeeping_features(df):
    x=df.copy(); x["date"]=pd.to_datetime(x.date); x["day_of_week"]=x.date.dt.day_name()
    x["checkout_flag"]=(x.cleaning_type=="Checkout").astype(int)
    return x

def add_stewarding_features(df):
    x=df.copy(); x["date"]=pd.to_datetime(x.date); x["day_of_week"]=x.date.dt.day_name()
    return x

def quality_risk_label(prob):
    return np.select([prob<.12,prob<.30],["LOW","MEDIUM"],default="HIGH")

def bottleneck_risk_label(util,wait):
    score=.7*np.asarray(util)+.3*np.clip(np.asarray(wait)/30,0,1.5)
    return np.select([score<.65,score<.90],["LOW","MEDIUM"],default="HIGH")
