"""One-shot setup: init DB -> load/generate data -> train models -> score shipments -> seed ledger.

Run:  python -m src.bootstrap [--data path/to/DataCoSupplyChainDataset.csv] [--max-rows 20000]
"""
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np

from . import config as cfg
from . import database as db
from .blockchain import Ledger
from .data_preprocessing import load_dataset
from .predict_risk import predict_frame
from .train_model import train

_PRODUCTS = [("Smartphone Lot", "Electronics", "Acme Devices"), ("Running Shoes", "Apparel", "StrideWorks"),
             ("Office Chairs", "Furniture", "ComfortCo"), ("Vaccine Vials", "Pharma Supplies", "MediPure"),
             ("Brake Pads", "Auto Parts", "TorqueParts"), ("Board Games", "Toys", "PlayMakers"),
             ("Laptop Batteries", "Electronics", "Voltix"), ("Canned Goods", "Groceries", "FreshLine"),
             ("Tennis Rackets", "Sporting Goods", "AceSport"), ("Winter Jackets", "Apparel", "NorthThread"),
             ("Standing Desks", "Furniture", "ErgoWorks"), ("Bike Helmets", "Sporting Goods", "SafeRide")]
_ROUTE = [("MANUFACTURED", "Factory", "Shenzhen, CN"), ("DISPATCHED", "Carrier", "Shenzhen Port, CN"),
          ("RECEIVED", "Warehouse", "Rotterdam DC, NL"), ("INSPECTED", "QA Team", "Rotterdam DC, NL"),
          ("DELIVERED", "Retailer", "Berlin, DE")]


def _seed_traceability(conn, shipments_df, rng):
    ledger = Ledger(conn)
    start = datetime.now(timezone.utc) - timedelta(days=30)
    batch_ids = []
    for i, (name, cat, supplier) in enumerate(_PRODUCTS, 1):
        bid = f"BATCH-{2026000 + i}"
        qty = int(rng.integers(200, 2000))
        db.create_batch(conn, bid, name, cat, qty, supplier.upper().replace(" ", "-"), "Shenzhen, CN")
        n_events = 5 if i % 3 else int(rng.integers(2, 5))  # a few batches are still in transit
        t = start + timedelta(days=int(rng.integers(0, 10)))
        for step in range(n_events):
            etype, role, loc = _ROUTE[step]
            who = supplier.upper().replace(" ", "-") if step == 0 else f"{role.upper()}-{int(rng.integers(100, 999))}"
            ledger.add_event(bid, etype, who, loc, qty, (t + timedelta(days=step * 2, hours=int(rng.integers(0, 12)))).isoformat(timespec="seconds"))
        batch_ids.append(bid)
    ledger.save_checkpoint("Initial demo seed")
    return batch_ids


def run_setup(data_path=None, max_rows=20000, progress=None):
    say = progress or (lambda msg: None)
    rng = np.random.default_rng(42)
    say("Loading and cleaning data...")
    df, source, report = load_dataset(data_path)
    say(f"Training models on {len(df):,} rows ({source})...")
    metrics, test_df = train(df, source, cleaning_report=report)
    say("Scoring held-out shipments...")
    ship = test_df.sample(min(max_rows, len(test_df)), random_state=1).reset_index(drop=True)
    ship = predict_frame(ship)
    in_transit = ship.index >= int(len(ship) * 0.9)  # newest 10% treated as not-yet-delivered: outcome hidden
    ship["status"] = np.where(in_transit, "In Transit", "Delivered")
    ship.loc[in_transit, ["late_delivery", "actual_days"]] = np.nan
    ship["batch_id"] = None
    ship["data_source"] = source
    say("Writing database and seeding the ledger...")
    conn = db.get_conn()
    try:
        db.init_db(conn, reset=True)
        batch_ids = _seed_traceability(conn, ship, rng)
        link = rng.random(len(ship)) < 0.3
        ship.loc[link, "batch_id"] = rng.choice(batch_ids, int(link.sum()))
        db.insert_shipments(conn, ship)
    finally:
        conn.close()
    say("Done.")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to DataCoSupplyChainDataset.csv (optional)")
    ap.add_argument("--max-rows", type=int, default=20000)
    a = ap.parse_args()
    m = run_setup(a.data, a.max_rows, progress=print)
    print("Random Forest:", m["random_forest"])
    print("Now run:  streamlit run app.py")
