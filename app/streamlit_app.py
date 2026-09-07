from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.models import train_housekeeping_models,train_stewarding_models
from src.preprocessing import quality_risk_label,bottleneck_risk_label

st.set_page_config(page_title="Hotel Operations Analytics",page_icon="🏨",layout="wide")
st.markdown("""<style>.block-container{padding-top:1.5rem;max-width:1250px}.kpi-note{color:#64748b;font-size:.85rem}</style>""",unsafe_allow_html=True)
st.title("Hotel Operations Analytics")
st.caption("SYNTHETIC PORTFOLIO PROJECT • Production Engineering + Data Analytics • No real hotel performance claims")

@st.cache_data
def load():
    hk=pd.read_csv(ROOT/'data/synthetic/housekeeping_operations.csv',parse_dates=['date'])
    sw=pd.read_csv(ROOT/'data/synthetic/stewarding_operations.csv',parse_dates=['date'])
    return hk,sw
@st.cache_resource
def models(hk,sw): return train_housekeeping_models(hk),train_stewarding_models(sw)

hk,sw=load(); hkm,swm=models(hk,sw)
page=st.sidebar.radio("Navigation",["Housekeeping","Stewarding","Continuous Improvement","Model Performance"])
st.sidebar.info("All KPIs, predictions and scenarios displayed here use synthetic data.")

if page=="Housekeeping":
    st.header("Housekeeping Process Optimization")
    day=st.date_input("Operational date",hk.date.max().date())
    d=hk[hk.date.dt.date==day]
    if d.empty: st.warning("Choose a date inside the synthetic dataset."); st.stop()
    staff=st.slider("Available room attendants",3,18,8); productive_hours=st.slider("Productive hours per attendant",4.0,8.0,6.0,.25)
    hours=d.cleaning_time_minutes.sum()/60; capacity=staff*productive_hours; gap=hours-capacity
    cols=st.columns(5)
    vals=[("Occupancy",f"{d.occupancy_rate.mean():.0%}"),("Rooms",len(d)),("Cleaning hours",f"{hours:.1f}"),("Required staff",int(np.ceil(hours/productive_hours))),("Capacity gap",f"{gap:+.1f} h")]
    for c,(a,b) in zip(cols,vals): c.metric(a,b)
    st.caption("Required staff is a synthetic planning estimate based on predicted productive hours, not a staffing standard.")
    reg,clf,_,_=hkm
    reg_cols=["room_type","cleaning_type","season","day_of_week","number_of_guests","occupancy_rate","cleaning_product_ml"]
    qc_cols=["room_type","cleaning_type","season","housekeeper_team","number_of_guests","cleaning_time_minutes","cleaning_product_ml","occupancy_rate"]
    d=d.copy(); d['predicted_cleaning_minutes']=reg.predict(d[reg_cols]).round(1)
    prob=clf.predict_proba(d[qc_cols])[:,1]; d['quality_risk_probability']=prob.round(3); d['quality_risk']=quality_risk_label(prob)
    a,b=st.columns(2)
    with a:
        st.subheader("Readiness and workload")
        st.bar_chart(d.groupby('room_type').cleaning_time_minutes.mean())
        st.write(f"Ready by 13:00: **{d.ready_by_13.mean():.1%}**")
        st.write(f"Re-clean rate: **{d.reclean_required.mean():.1%}**")
    with b:
        st.subheader("Supervisor priority list")
        st.dataframe(d.sort_values(['quality_risk_probability','predicted_cleaning_minutes'],ascending=False)[['room_number','room_type','cleaning_type','housekeeper_team','predicted_cleaning_minutes','quality_risk']],hide_index=True,use_container_width=True)
    st.info("Potential actions: rebalance high-time rooms across teams, stage linen before peak departures, and inspect high-risk rooms first. These are proposed actions, not measured improvements.")

elif page=="Stewarding":
    st.header("Stewarding & Dishwashing Process Optimization")
    day=st.date_input("Operational date",sw.date.max().date()); meal=st.selectbox("Meal period",["Breakfast","Lunch","Dinner"])
    d=sw[(sw.date.dt.date==day)&(sw.meal_period==meal)]
    if d.empty: st.warning("Choose a date inside the synthetic dataset."); st.stop()
    items=d.number_of_items.sum(); racks=d.dishwasher_cycles.sum(); hours=d.washing_time_minutes.sum()/60
    utilisation=d.dishwasher_utilization.mean(); wait=d.waiting_time_minutes.mean(); risk=bottleneck_risk_label([utilisation],[wait])[0]
    cols=st.columns(5)
    vals=[("Covers",int(d.restaurant_covers.max())),("Expected items",int(items)),("Expected racks",int(racks)),("Wash hours",f"{hours:.1f}"),("Bottleneck risk",risk)]
    for c,(a,b) in zip(cols,vals): c.metric(a,b)
    a,b=st.columns(2)
    with a:
        st.subheader("Flow and capacity")
        st.metric("Dishwasher utilisation",f"{utilisation:.0%}"); st.metric("Average waiting",f"{wait:.1f} min")
        st.bar_chart(d.set_index('dish_category').number_of_items)
    with b:
        st.subheader("Quality and resources")
        st.metric("Rewash/reject rate",f"{d.items_rejected.sum()/items:.1%}"); st.metric("Throughput",f"{items/max(hours,0.1):.0f} items/wash-hour")
        st.dataframe(d[['dish_category','number_of_items','dishwasher_cycles','waiting_time_minutes','water_consumption_liters','detergent_consumption_ml']],hide_index=True,use_container_width=True)
    st.info("Potential actions: pre-sort categories, stage correctly loaded racks, move support to the peak receiving window, and investigate high rewash together with utilisation. Hygiene remains a constraint, never a trade-off.")

elif page=="Continuous Improvement":
    st.header("Continuous Improvement • PLAN → DO → CHECK → ACT")
    st.markdown("Use one bounded test at a time. A result is recorded only after observation; blank fields below deliberately avoid fake improvement claims.")
    template=pd.DataFrame([
        ["Late checkout rooms","Ready by 13:00","Synthetic baseline from selected week","Stage linen by floor; priority sequence","Not measured","Collect one comparable week"],
        ["Dinner rack queue","Average waiting minutes","Synthetic baseline from selected week","Pre-sort and assign peak runner","Not measured","Check waiting and rewash together"],
    ],columns=["Problem","KPI","Baseline","Intervention","Result","Next action"])
    st.data_editor(template,num_rows="dynamic",use_container_width=True)
    st.subheader("Weekly control plan")
    st.dataframe(pd.DataFrame({"Area":["Housekeeping","Housekeeping","Housekeeping","Stewarding","Stewarding","Stewarding"],"KPI":["Cleaning minutes by type","Inspection/re-clean rate","Ready by 13:00","Items per hour","Waiting and utilisation","Rewash, water, detergent"],"Supervisor question":["Is mix or process driving time?","Is speed harming quality?","Which arrival/departure window failed?","Did output match demand?","Where did the queue form?","Did resource use rise without quality gain?"]}),hide_index=True,use_container_width=True)

else:
    st.header("Model Performance")
    _,_,hr,hc=hkm; _,_,sr,sc,_=swm
    st.warning("Metrics are hold-out results on synthetic data. They demonstrate workflow, not expected performance in a real hotel.")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Housekeeping")
        st.write("Cleaning-time regression",pd.DataFrame([hr]).round(3))
        st.write("Inspection-risk classification",pd.DataFrame([{k:v for k,v in hc.items() if k!='Confusion matrix'}]).round(3))
        st.write("Confusion matrix",pd.DataFrame(hc['Confusion matrix'],index=['Actual pass','Actual fail'],columns=['Predicted pass','Predicted fail']))
    with c2:
        st.subheader("Stewarding")
        st.write("Items regression",pd.DataFrame([sr]).round(3))
        st.write("Bottleneck classification",pd.DataFrame([{k:v for k,v in sc.items() if k!='Confusion matrix'}]).round(3))
        st.write("Confusion matrix",pd.DataFrame(sc['Confusion matrix'],index=['Actual normal','Actual bottleneck'],columns=['Predicted normal','Predicted bottleneck']))
    st.markdown("**Operational meaning:** MAE expresses typical planning error in minutes or items. Recall matters when missing a quality or bottleneck risk is costly. Precision matters because too many false alerts waste supervisor attention. Models support prioritisation and capacity planning; supervisors retain the decision.")
