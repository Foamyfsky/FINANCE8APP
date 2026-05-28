"""
generate_splits.py

Generates the shared train/test dataset for the fair model comparison.

Outputs -> DATA3888G08/splits/
  train.csv        - buckets 1-16,  60 stocks, 300 time_ids
  test.csv         - buckets 17-20, 60 stocks, 300 time_ids
  stock_meta.csv   - stock_id, regime, median_bas, median_rv

Columns: stock_id, time_id, time_bucket, wap, bas, rv, regime
"""

import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

INPUT_CSV = r"D:\USYD\DATA3888\group_asm\optiver_aggregated.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "splits")
os.makedirs(OUT_DIR, exist_ok=True)

# Exact stock lists and time_ids from Jamie's shared config
LIQUID_STOCKS = [
    10, 13, 20, 28, 39, 46, 47, 50, 51, 52,
    64, 68, 69, 77, 85, 86, 95, 99, 120, 124,
]
MIXED_STOCKS = [
    7, 19, 21, 26, 32, 42, 48, 59, 61, 70,
    73, 76, 82, 89, 96, 101, 107, 113, 114, 115,
]
ILLIQUID_STOCKS = [
    0, 3, 4, 6, 9, 23, 30, 37, 38, 40,
    60, 80, 81, 88, 97, 98, 102, 103, 112, 118,
]
ALL_STOCKS = sorted(LIQUID_STOCKS + MIXED_STOCKS + ILLIQUID_STOCKS)
REGIME_MAP = (
    {s: "liquid"   for s in LIQUID_STOCKS}
  | {s: "mixed"    for s in MIXED_STOCKS}
  | {s: "illiquid" for s in ILLIQUID_STOCKS}
)

SAMPLED_TIME_IDS = [
    256, 650, 745, 748, 785, 1070, 1178, 1227, 1359, 1820, 1904, 2006,
    2037, 2185, 2331, 2410, 2444, 2525, 2553, 2593, 2631, 2656, 2659,
    2867, 2893, 2917, 3146, 3152, 3406, 3607, 3721, 3884, 3921, 4004,
    4158, 4173, 4364, 4493, 4560, 4627, 4723, 4743, 4850, 4905, 5032,
    5063, 5305, 5340, 5458, 5470, 5598, 5620, 5743, 5825, 5829, 5916,
    6177, 6476, 6626, 6693, 6979, 7041, 7219, 7270, 8081, 8191, 8196,
    8256, 8448, 8534, 8583, 8585, 8665, 8721, 8763, 9077, 9153, 9302,
    9352, 9367, 9456, 9564, 9700, 9862, 9918, 10017, 10042, 10488,
    10604, 10702, 10725, 10776, 10781, 10994, 11090, 11243, 11375,
    11508, 11869, 11940, 12030, 12420, 12579, 12581, 12676, 12834,
    12957, 12991, 13180, 13235, 13269, 13362, 13382, 13410, 13421,
    13432, 13451, 13452, 13598, 13821, 13900, 13941, 13960, 13986,
    14001, 14004, 14077, 14125, 14235, 14246, 14247, 14278, 14279,
    14311, 14465, 14752, 14769, 14810, 14976, 15061, 15209, 15276,
    15280, 15319, 15341, 15400, 15448, 15499, 15516, 15529, 15547,
    16086, 16288, 16438, 16570, 16802, 16816, 16910, 17013, 17056,
    17058, 17157, 17169, 17172, 17191, 17264, 17398, 17983, 18012,
    18180, 18200, 18218, 18400, 18509, 18595, 18629, 18792, 18916,
    19065, 19072, 19180, 19226, 19385, 19499, 19554, 19678, 19994,
    20063, 20172, 20199, 20317, 20418, 20457, 20469, 20673, 20691,
    20732, 20972, 21079, 21083, 21208, 21272, 21428, 21445, 21685,
    21734, 22011, 22120, 22217, 22304, 22392, 22519, 22622, 22635,
    22750, 22829, 23030, 23185, 23228, 23232, 23337, 23642, 23708,
    23819, 23823, 23858, 23903, 24157, 24179, 24388, 24393, 24396,
    24473, 24535, 24816, 24817, 25019, 25087, 25312, 25318, 25335,
    25369, 25389, 25488, 25501, 25599, 25654, 25854, 26208, 26337,
    26447, 26568, 26844, 27020, 27278, 27304, 27313, 27711, 27868,
    28070, 28512, 28541, 28634, 28697, 29316, 29507, 29570, 29592,
    29853, 30412, 30454, 30527, 30598, 30748, 30790, 30816, 30896,
    31055, 31083, 31119, 31138, 31327, 31412, 31482, 31522, 31656,
    32126, 32186, 32200, 32249, 32277, 32322, 32376, 32463, 32590,
    32653, 32662, 32680, 32746, 32748,
]

TRAIN_BUCKETS = list(range(1, 17))   # 1-16
TEST_BUCKETS = list(range(17, 21))   # 17-20

print("Loading data ...")
df = (pd.read_csv(INPUT_CSV)
        .rename(columns={"WAP_mean":          "wap",
                         "BidAskSpread_mean":  "bas",
                         "volatility":         "rv"})
        .query("time_bucket > 0")
        .query("stock_id in @ALL_STOCKS")
        .query("time_id   in @SAMPLED_TIME_IDS")
        .sort_values(["stock_id", "time_id", "time_bucket"])
        .reset_index(drop=True))

df["regime"] = df["stock_id"].map(REGIME_MAP)

# Keep only sessions with all 20 buckets present
counts = df.groupby(["stock_id", "time_id"])["time_bucket"].count()
complete = counts[counts == 20].index
df = df.set_index(["stock_id", "time_id"]).loc[complete].reset_index()

print(f"  {len(df):,} rows | {df['stock_id'].nunique()} stocks "
      f"| {df['time_id'].nunique()} time_ids")
print(f"  Regimes: liquid={df[df.regime=='liquid']['stock_id'].nunique()} stocks  "
      f"mixed={df[df.regime=='mixed']['stock_id'].nunique()} stocks  "
      f"illiquid={df[df.regime=='illiquid']['stock_id'].nunique()} stocks")

train = df[df["time_bucket"].isin(TRAIN_BUCKETS)].copy()
test = df[df["time_bucket"].isin(TEST_BUCKETS)].copy()

print(f"\nTrain: {len(train):,} rows  ({train['time_bucket'].nunique()} buckets: "
      f"{train['time_bucket'].min()}-{train['time_bucket'].max()})")
print(f"Test : {len(test):,} rows  ({test['time_bucket'].nunique()} buckets: "
      f"{test['time_bucket'].min()}-{test['time_bucket'].max()})")

stock_meta = (train.groupby(["stock_id", "regime"])
              .agg(median_bas=("bas", "median"),
                   median_rv =("rv",  "median"),
                   n_sessions=("time_id", "nunique"))
              .reset_index()
              .sort_values(["regime", "stock_id"]))

train_path = os.path.join(OUT_DIR, "train.csv")
test_path = os.path.join(OUT_DIR, "test.csv")
meta_path = os.path.join(OUT_DIR, "stock_meta.csv")

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)
stock_meta.to_csv(meta_path, index=False)

print(f"\nSaved:")
print(f"  {train_path}  ({os.path.getsize(train_path)//1024:,} KB)")
print(f"  {test_path}   ({os.path.getsize(test_path)//1024:,} KB)")
print(f"  {meta_path}")

print(f"\nStock meta preview:")
print(stock_meta.to_string(index=False))
