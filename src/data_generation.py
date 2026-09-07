"""Deterministic synthetic hotel operations data.

No row represents a real hotel, employee or measured result. Distributions are modelling
assumptions chosen to make operational analysis realistic and interpretable.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOMS = pd.DataFrame({
    "room_number": list(range(101,141))+list(range(201,241))+list(range(301,321)),
    "room_type": ["Standard"]*40+["Superior"]*40+["Suite"]*20,
})
TEAMS=["Team A","Team B","Team C","Team D","Team E","Team F"]


def season_of(month):
    return "Winter" if month in [12,1,2] else "Spring" if month in [3,4,5] else "Summer" if month in [6,7,8] else "Autumn"


def generate_housekeeping(start="2025-01-01", months=9, seed=42):
    rng=np.random.default_rng(seed)
    dates=pd.date_range(start, periods=int(months*30.4), freq="D")
    rows=[]
    occ_effect={"Winter":.87,"Spring":.72,"Summer":.82,"Autumn":.68}
    base_time={"Standard":24,"Superior":30,"Suite":43}
    product_base={"Standard":95,"Superior":120,"Suite":175}
    for date in dates:
        season=season_of(date.month)
        weekend=date.dayofweek>=4
        p=np.clip(occ_effect[season]+(.05 if weekend else 0)+rng.normal(0,.05),.45,.98)
        occupied=rng.random(len(ROOMS))<p
        for idx,room in ROOMS[occupied].iterrows():
            checkout=rng.random()<(.31 if weekend else .25)
            cleaning="Checkout" if checkout else "Stayover"
            guests=int(rng.integers(1,3 if room.room_type=="Standard" else 5))
            team=TEAMS[(room.room_number+date.day)%len(TEAMS)]
            minutes=base_time[room.room_type]+(10 if checkout else -5)+1.7*(guests-1)+(3 if season=="Winter" else 0)+rng.normal(0,4)
            minutes=max(10,round(minutes,1))
            # U-shaped quality risk: rushing and extreme overruns may both signal process difficulty.
            expected=base_time[room.room_type]+(10 if checkout else -5)
            quality_logit=-2.6+.13*max(expected-minutes,0)+.04*max(minutes-expected-8,0)+(.55 if checkout else 0)
            fail=rng.random()<1/(1+np.exp(-quality_logit))
            reclean=fail and rng.random()<.72
            product=max(30,round(product_base[room.room_type]*(1.18 if checkout else .72)+9*(guests-1)+rng.normal(0,14),1))
            start_min=8*60+int(rng.integers(0,210))
            ready_min=int(start_min+minutes+(12 if reclean else 0)+rng.integers(3,18))
            rows.append({
                "date":date.date(),"room_number":room.room_number,"room_type":room.room_type,
                "number_of_guests":guests,"check_in":checkout,"check_out":checkout,"stayover":not checkout,
                "occupancy_rate":round(p,3),"season":season,"day_of_week":date.day_name(),
                "cleaning_type":cleaning,"cleaning_time_minutes":minutes,"housekeeper_team":team,
                "inspection_result":"Fail" if fail else "Pass","reclean_required":bool(reclean),
                "cleaning_product_ml":product,"room_readiness_time":f"{ready_min//60:02d}:{ready_min%60:02d}",
                "ready_by_13":ready_min<=13*60,
            })
    return pd.DataFrame(rows)


def generate_stewarding(start="2025-01-01", months=9, seed=84):
    rng=np.random.default_rng(seed)
    dates=pd.date_range(start, periods=int(months*30.4), freq="D")
    rows=[]
    cover_base={"Breakfast":105,"Lunch":70,"Dinner":125}
    categories={"Plates":2.2,"Glassware":1.4,"Cutlery":2.8,"Cookware":.22}
    for date in dates:
        season=season_of(date.month); weekend=date.dayofweek>=4
        occupancy=np.clip(({"Winter":.87,"Spring":.72,"Summer":.82,"Autumn":.68}[season])+rng.normal(0,.05),.45,.98)
        event=bool(rng.random()<(.14 if weekend else .07))
        for meal in cover_base:
            covers=max(20,int(cover_base[meal]*occupancy+(28 if event and meal=="Dinner" else 0)+rng.normal(0,10)))
            staff=max(2,int(round(2+covers/85+(1 if event else 0))))
            peak=meal=="Dinner" or event
            for category,mult in categories.items():
                items=max(5,int(covers*mult+rng.normal(0,max(4,covers*.08))))
                per_rack={"Plates":24,"Glassware":20,"Cutlery":45,"Cookware":10}[category]
                racks=int(np.ceil(items/per_rack)); cycle=2.1+(0.35 if category=="Cookware" else 0)+rng.normal(0,.13)
                wash=racks*cycle; capacity_items=per_rack*60/cycle
                required_rate=items/(1.35 if meal=="Dinner" else 1.7)
                util=np.clip(4.2*required_rate/capacity_items,0.25,1.25)
                wait=max(0,round(3+34*max(util-.72,0)+(8 if event else 0)+rng.normal(0,3),1))
                rewash_prob=np.clip(.018+.035*max(util-.82,0)+(.025 if category=="Cookware" else 0),.005,.13)
                rejected=int(rng.binomial(items,rewash_prob)); damaged=int(rng.binomial(items,.0015))
                water=round(racks*(3.8+rng.normal(0,.22))+items*.08,1)
                detergent=round(racks*(10+rng.normal(0,1.0)),1)
                rows.append({"date":date.date(),"meal_period":meal,"restaurant_covers":covers,
                    "event_indicator":event,"occupancy_rate":round(occupancy,3),"number_of_items":items,
                    "dish_category":category,"dishwasher_cycles":racks,"cycle_duration_minutes":round(cycle,2),
                    "items_per_rack":per_rack,"washing_time_minutes":round(wash,1),"waiting_time_minutes":wait,
                    "rewash_required":rejected>0,"items_rejected":rejected,"damaged_items":damaged,
                    "dishwasher_utilization":round(util,3),"staff_available":staff,
                    "water_consumption_liters":water,"detergent_consumption_ml":detergent,
                    "season":season,"day_of_week":date.day_name(),"peak_period":peak})
    return pd.DataFrame(rows)


def save_synthetic(output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    hk=generate_housekeeping(); st=generate_stewarding()
    hk.to_csv(out/'housekeeping_operations.csv',index=False)
    st.to_csv(out/'stewarding_operations.csv',index=False)
    return hk,st

if __name__=="__main__": save_synthetic(Path(__file__).parents[1]/"data"/"synthetic")
