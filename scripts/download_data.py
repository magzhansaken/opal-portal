"""Download OULAD (UCI Machine Learning Repository) and act-mooc (SNAP) into data/ in the layout used by
opal.data.  OULAD CSV files are converted to Parquet, and the click table is partitioned by module and
presentation.  Row counts are checked against the published dataset descriptions.

usage: python scripts/download_data.py [--oulad-zip PATH] [--mooc-tar PATH]
Use the --*-zip/--*-tar options when the archives were downloaded manually.
"""
import argparse
import io
import os
import tarfile
import urllib.request
import zipfile

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OULAD_URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
MOOC_URL = "https://snap.stanford.edu/data/act-mooc.tar.gz"
EXPECTED = {"studentInfo": 32593, "studentRegistration": 32593, "courses": 22, "assessments": 206,
            "studentAssessment": 173912, "vle": 6364, "studentVle": 10655280}
NUMERIC = ["id_student", "id_site", "id_assessment", "date", "date_registration", "date_unregistration",
           "date_submitted", "score", "weight", "sum_click", "num_of_prev_attempts", "studied_credits",
           "module_presentation_length", "is_banked", "week_from", "week_to"]


def fetch(url, path):
    if path and os.path.exists(path):
        return open(path, "rb").read()
    print("downloading", url)
    with urllib.request.urlopen(url) as r:
        return r.read()


def read_csv(zf, name):
    member = next(m for m in zf.namelist() if m.endswith(f"{name}.csv"))
    df = pd.read_csv(zf.open(member), na_values=["?", ""], keep_default_na=True)
    for c in df.columns:
        if c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def convert_oulad(raw, out):
    os.makedirs(out, exist_ok=True)
    outer = zipfile.ZipFile(io.BytesIO(raw))
    inner = [m for m in outer.namelist() if m.endswith(".zip")]
    zf = zipfile.ZipFile(io.BytesIO(outer.read(inner[0]))) if inner else outer
    for name in ["studentInfo", "studentRegistration", "courses", "assessments", "studentAssessment", "vle"]:
        df = read_csv(zf, name)
        assert len(df) == EXPECTED[name], (name, len(df))
        df.to_parquet(os.path.join(out, f"{name}.parquet"), index=False)
        print(f"{name}: {len(df):,} rows")
    svle = read_csv(zf, "studentVle")
    assert len(svle) == EXPECTED["studentVle"], len(svle)
    svle.to_parquet(os.path.join(out, "studentVle_ds"), partition_cols=["code_module", "code_presentation"], index=False)
    print(f"studentVle: {len(svle):,} rows")


def extract_mooc(raw, out):
    os.makedirs(out, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(".tsv"):
                with open(os.path.join(out, os.path.basename(m.name)), "wb") as f:
                    f.write(tf.extractfile(m).read())
    n = sum(1 for _ in open(os.path.join(out, "mooc_actions.tsv"))) - 1
    assert n == 411749, n
    print(f"act-mooc: {n:,} actions")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--oulad-zip"); p.add_argument("--mooc-tar")
    a = p.parse_args()
    convert_oulad(fetch(OULAD_URL, a.oulad_zip), os.path.join(ROOT, "oulad"))
    extract_mooc(fetch(MOOC_URL, a.mooc_tar), os.path.join(ROOT, "act-mooc"))
    print("done; the first call of opal.data.load() builds the cached cohorts")
