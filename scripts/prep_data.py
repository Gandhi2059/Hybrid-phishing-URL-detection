"""
scripts/prep_data.py
--------------------
Merge, deduplicate, and shuffle all raw CSV datasets into a single
canonical file: data/urls.csv

Expected raw files
------------------
data/phishing_site_urls.csv  — columns: URL, Label (good/bad)
data/urldata.csv             — columns: url, label (benign/malicious)

Output
------
data/urls.csv                — columns: url (str), label (0=legit, 1=phish)

Usage
-----
    python scripts/prep_data.py
"""

import sys
import os

# Allow importing src/ from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────── #
DATA_DIR          = os.path.join(os.path.dirname(__file__), "..", "data")
PHISHING_SITE_CSV = os.path.join(DATA_DIR, "phishing_site_urls.csv")
URLDATA_CSV       = os.path.join(DATA_DIR, "urldata.csv")
OUTPUT_CSV        = os.path.join(DATA_DIR, "urls.csv")

print("=" * 60)
print("  Data Preparation — PhishDetect")
print("=" * 60)

# ── Dataset 1: phishing_site_urls.csv ────────────────────────────────────── #
print("\n[1/2] Loading phishing_site_urls.csv …")
df1 = pd.read_csv(PHISHING_SITE_CSV)
df1 = df1.rename(columns={"URL": "url", "Label": "label"})
df1["label"] = df1["label"].map({"good": 0, "bad": 1})
df1 = df1[["url", "label"]].dropna()
print(f"      Rows: {len(df1):,}  |  Phishing: {df1['label'].sum():,}"
      f"  |  Legit: {(df1['label']==0).sum():,}")

# ── Dataset 2: urldata.csv ───────────────────────────────────────────────── #
print("\n[2/2] Loading urldata.csv …")
df2 = pd.read_csv(URLDATA_CSV, usecols=["url", "label"])
df2["label"] = df2["label"].map({"benign": 0, "malicious": 1})
df2 = df2[["url", "label"]].dropna()
print(f"      Rows: {len(df2):,}  |  Phishing: {df2['label'].sum():,}"
      f"  |  Legit: {(df2['label']==0).sum():,}")

# ── Merge & deduplicate ──────────────────────────────────────────────────── #
print("\nMerging and deduplicating …")
combined   = pd.concat([df1, df2], ignore_index=True)
n_before   = len(combined)
combined   = combined.drop_duplicates(subset="url", keep="first")
n_after    = len(combined)
combined   = combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Combined rows  : {n_before:,}")
print(f"  After dedup    : {n_after:,}  (removed {n_before - n_after:,})")

print("\n" + "=" * 60)
print(f"  FINAL DATASET : {n_after:,} rows")
print(f"  Legitimate (0): {(combined['label']==0).sum():,}"
      f"  ({(combined['label']==0).mean()*100:.1f}%)")
print(f"  Phishing   (1): {(combined['label']==1).sum():,}"
      f"  ({(combined['label']==1).mean()*100:.1f}%)")
print("=" * 60)

combined.to_csv(OUTPUT_CSV, index=False)
print(f"\n✔ Saved combined dataset → {OUTPUT_CSV}")
