"""Generates a clearly-labelled SYNTHETIC shipment dataset (used when DataCo is not downloaded).

The late-delivery probability is driven by a made-up logistic formula. Models trained
on this data only demonstrate the pipeline; their scores say nothing about the real world.
"""
import numpy as np
import pandas as pd

MODES = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}
REGIONS = {"Western Europe": "Europe", "Northern Europe": "Europe", "Central America": "LATAM",
           "South America": "LATAM", "US Center": "USCA", "Canada": "USCA",
           "Southeast Asia": "Pacific Asia", "South Asia": "Pacific Asia",
           "Oceania": "Pacific Asia", "West Africa": "Africa"}
CATEGORIES = ["Electronics", "Apparel", "Sporting Goods", "Furniture", "Toys",
              "Pharma Supplies", "Auto Parts", "Groceries"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
_MODE_EFFECT = {"Standard Class": -0.3, "Second Class": 0.4, "First Class": 1.6, "Same Day": 0.6}


def generate_shipments(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region_names = list(REGIONS)
    region_effect = dict(zip(region_names, np.random.default_rng(1).normal(0, 0.4, len(region_names))))
    mode = rng.choice(list(MODES), n, p=[0.55, 0.2, 0.15, 0.10])
    region = rng.choice(region_names, n)
    qty = rng.integers(1, 6, n)
    price = np.round(rng.uniform(10, 500, n), 2)
    disc = np.round(rng.choice([0, 0.05, 0.1, 0.15, 0.2, 0.25], n), 2)
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    month = dates.month.to_numpy()
    logit = (-1.0 + np.array([_MODE_EFFECT[m] for m in mode]) + np.array([region_effect[r] for r in region])
             + 0.15 * np.log1p(qty) + (month >= 10) * 0.5 - 0.8 * disc + rng.normal(0, 0.6, n))
    late = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    sched = np.array([MODES[m] for m in mode])
    actual = np.where(late == 1, sched + rng.integers(1, 4, n), np.maximum(sched - rng.integers(0, 2, n), 0))
    return pd.DataFrame({
        "shipment_id": [f"SHP-{i:06d}" for i in range(1, n + 1)],
        "order_date": dates.strftime("%Y-%m-%d"),
        "shipping_mode": mode, "scheduled_days": sched, "region": region,
        "market": [REGIONS[r] for r in region], "category": rng.choice(CATEGORIES, n),
        "quantity": qty, "unit_price": price, "discount_rate": disc,
        "customer_segment": rng.choice(SEGMENTS, n), "late_delivery": late, "actual_days": actual,
    })
