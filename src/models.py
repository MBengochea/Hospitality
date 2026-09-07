import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from .metrics import regression_metrics, classification_metrics

def _pipe(cat,num,model):
    prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num","passthrough",num)])
    return Pipeline([("prep",prep),("model",model)])

def train_housekeeping_models(df):
    cat=["room_type","cleaning_type","season","day_of_week"]
    num=["number_of_guests","occupancy_rate","cleaning_product_ml"]
    X=df[cat+num]; y=df.cleaning_time_minutes
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42)
    reg=_pipe(cat,num,RandomForestRegressor(n_estimators=120,min_samples_leaf=4,random_state=42,n_jobs=-1)).fit(Xt,yt)
    reg_m=regression_metrics(yv,reg.predict(Xv))
    qc_cat=["room_type","cleaning_type","season","housekeeper_team"]
    qc_num=["number_of_guests","cleaning_time_minutes","cleaning_product_ml","occupancy_rate"]
    X=df[qc_cat+qc_num]; y=(df.inspection_result=="Fail").astype(int)
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    clf=_pipe(qc_cat,qc_num,RandomForestClassifier(n_estimators=150,class_weight="balanced",min_samples_leaf=5,random_state=42,n_jobs=-1)).fit(Xt,yt)
    pred=clf.predict(Xv); clf_m=classification_metrics(yv,pred)
    return reg,clf,reg_m,clf_m

def train_stewarding_models(df):
    daily=df.groupby(["date","meal_period","season","day_of_week","event_indicator"],as_index=False).agg(
        restaurant_covers=("restaurant_covers","max"),occupancy_rate=("occupancy_rate","max"),number_of_items=("number_of_items","sum"),
        staff_available=("staff_available","max"),waiting_time_minutes=("waiting_time_minutes","mean"),dishwasher_utilization=("dishwasher_utilization","mean"))
    cat=["meal_period","season","day_of_week","event_indicator"]; num=["restaurant_covers","occupancy_rate","staff_available"]
    X=daily[cat+num]; y=daily.number_of_items
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42)
    reg=_pipe(cat,num,RandomForestRegressor(n_estimators=120,min_samples_leaf=3,random_state=42,n_jobs=-1)).fit(Xt,yt)
    reg_m=regression_metrics(yv,reg.predict(Xv))
    daily["bottleneck"]=(daily.dishwasher_utilization>.78)|(daily.waiting_time_minutes>12)
    X=daily[cat+num]; y=daily.bottleneck.astype(int)
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    clf=_pipe(cat,num,RandomForestClassifier(n_estimators=150,class_weight="balanced",min_samples_leaf=4,random_state=42,n_jobs=-1)).fit(Xt,yt)
    clf_m=classification_metrics(yv,clf.predict(Xv))
    return reg,clf,reg_m,clf_m,daily
