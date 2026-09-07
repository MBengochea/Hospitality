# Hotel Operations Analytics

> **Portfolio disclaimer:** This repository applies Data Analysis to hotel operations. It does not claim professional hotel-management experience. Every operational row, KPI, model result and scenario is synthetic or modelled.

## Business problem
Housekeeping and stewarding supervisors must match daily demand to limited people and equipment while protecting readiness, hygiene, quality and resource control. This repository demonstrates a practical measurement and decision-support approach.

## Operational context
**Housekeeping:** room demand varies by occupancy, departures, room type and season. **Stewarding:** dish flow varies by covers, meal period, events, category, rack loading and machine capacity.

## Dataset
Deterministic synthetic generators create nine months of room-level and dish-category/shift-level records. Names are generic teams, not people. Assumptions are intentionally transparent and modest. Rebuild with `python -m src.data_generation`.

## Methodology
1. Measure operational events and resources.
2. Analyze workload, capacity, queues, quality and consumption.
3. Predict cleaning time, workload and operational risk.
4. Propose actions, without claiming achieved improvement.
5. Monitor weekly KPIs using a PDCA log.

## Analysis and KPIs
- Housekeeping: minutes per room, checkout/stayover mix, readiness by 13:00, inspection-failure and re-clean rates, product use and team workload.
- Stewarding: items/hour, rack cycles, utilisation, waiting, rewash/rejects, damaged items, water and detergent use.

## Machine-learning approach
Interpretable tabular pipelines use one-hot encoding and random forests. A fixed train/test split reports MAE, RMSE and R² for regression, plus accuracy, precision, recall, F1 and confusion matrices for classification. Risk classes support prioritisation; they do not replace supervisor inspection or hygiene controls.

## Operational recommendations
The app translates forecast demand into required labour hours, capacity gaps, priority rooms and shift bottleneck risks. Recommendations are possible actions such as rebalancing floors, staging linen/racks, improving sorting or scheduling peak support.

## Limitations
- No real hotel data or measured business impact.
- Synthetic relationships can make models look cleaner than production data.
- Readiness, hygiene and staffing rules differ by property, equipment and agreement.
- Models require validation, bias review, workflow integration and human oversight before operational use.

## Continuous improvement
Use weekly PDCA reviews: define one problem and KPI, preserve a baseline, test one bounded intervention, check quality and workload together, then standardise only when repeat measurements support it.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_generation
streamlit run app/streamlit_app.py
```

## Repository structure
```text
hotel-operations-analytics/
├── projects/01_housekeeping/
├── projects/02_stewarding/
├── notebooks/
├── src/
├── app/streamlit_app.py
├── data/synthetic/
└── requirements.txt
```

## Relevance for Swiss employers
The portfolio treats productivity indicators as planning tools, not worker quotas. External references used only for context include the Swiss hospitality collective agreement and general housekeeping/dishwasher benchmarks. Property-specific standards should always replace assumptions.
