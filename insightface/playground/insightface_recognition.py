#!/usr/bin/env python

import os
import sys
import cv2
import csv
import threading
import tqdm
import concurrent.futures
import multiprocessing
import numpy as np
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} probes_manifest.csv outfile.csv")
    sys.exit(1)

probes_file = sys.argv[1]
outfile = sys.argv[2]

GPUS = [0]
WORKERS = [0]
local = threading.local()

# ------------------------------
# ThreadPool initializer
# ------------------------------
def initializer(q, local):
    [local.g, local.w] = q.get()
    provider = [('CUDAExecutionProvider', {'device_id': local.g})]
    local.app = FaceAnalysis(providers=provider, allowed_modules=['detection', 'recognition'])
    local.app.prepare(ctx_id=local.w, det_size=(512,512))

# ------------------------------
# Load probes manifest
# ------------------------------
probes_data = []
with open(probes_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        probes_data.append({
            "identity": row["identity"],
            "step": row["step"],
            "radius": float(row["radius"]),
            "image_file": row["image_file"],
            "z_file": row["z_file"],
            "w_file": row["w_file"]
        })

# ------------------------------
# Build reference embeddings (radius 0)
# ------------------------------
print("Building reference embeddings...")
reference_embeddings = {}
provider = [('CUDAExecutionProvider', {'device_id': 0})]
app = FaceAnalysis(providers=provider, allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(512,512))

for row in probes_data:
    if row["step"] == "0" and row["radius"] == 0.0:
        img = cv2.imread(row["image_file"])
        if img is None:
            continue
        faces = app.get(img)
        if len(faces) > 0:
            reference_embeddings[row["identity"]] = faces[0].normed_embedding

print(f"Got {len(reference_embeddings)} reference embeddings.")

# ------------------------------
# Worker function
# ------------------------------
def worker(row, local, threshold=0.8):
    img = cv2.imread(row["image_file"])
    if img is None:
        return {"identity": row["identity"], "step": row["step"], 
                "radius": row["radius"], "face_detected": 0, "recognized": 0}

    faces = local.app.get(img)
    if len(faces) == 0:
        return {"identity": row["identity"], "step": row["step"], 
                "radius": row["radius"], "face_detected": 0, "recognized": 0}

    emb = faces[0].normed_embedding
    ref = reference_embeddings.get(row["identity"])
    if ref is None:
        return {"identity": row["identity"], "step": row["step"], 
                "radius": row["radius"], "face_detected": 1, "recognized": 0}

    sim = np.dot(emb, ref)  # cosine similarity
    recognized = int(sim > threshold)
    return {"identity": row["identity"], "step": row["step"], 
            "radius": row["radius"], "face_detected": 1, "recognized": recognized}

# ------------------------------
# Run in parallel
# ------------------------------
results = []
q = multiprocessing.Queue()
for g in GPUS:
    for w in WORKERS:
        q.put([g, w])

with concurrent.futures.ThreadPoolExecutor(
    max_workers=len(GPUS)*len(WORKERS),
    initializer=initializer,
    initargs=(q, local)
) as executor:
    futures = [executor.submit(worker, row, local) for row in probes_data]
    for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        results.append(f.result())

# ------------------------------
# Save detailed results
# ------------------------------
with open(outfile, "w", newline='') as f:
    fieldnames = ["identity", "step", "radius", "face_detected", "recognized"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Saved detailed results to {outfile}")

# ------------------------------
# Summarize per identity & radius
# ------------------------------
summary_dict = {}
for row in results:
    key = (row["identity"], row["radius"])
    summary_dict.setdefault(key, []).append(row)

summary = []
for (identity, radius), rows in summary_dict.items():
    det_rate = sum(r["face_detected"] for r in rows) / len(rows)
    rec_rate = sum(r["recognized"] for r in rows) / len(rows)
    summary.append({
        "identity": identity,
        "radius": radius,
        "detection_rate": det_rate,
        "recognition_rate": rec_rate
    })

summary_file = outfile.replace(".csv", "_summary.csv")
with open(summary_file, "w", newline='') as f:
    fieldnames = ["identity", "radius", "detection_rate", "recognition_rate"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary:
        writer.writerow(row)

print(f"Saved summary to {summary_file}")

# ------------------------------
# Plot recognition rate vs radius
# ------------------------------
plt.figure(figsize=(8,6))
identities = sorted(set(s["identity"] for s in summary))
for identity in identities:
    rows = sorted([s for s in summary if s["identity"] == identity], key=lambda x: x["radius"])
    radii = [r["radius"] for r in rows]
    rec_rates = [r["recognition_rate"] for r in rows]
    plt.plot(radii, rec_rates, marker="o", label=f"Identity {identity}")

plt.xlabel("Radius")
plt.ylabel("Recognition Rate")
plt.title("Recognition Rate vs Radius per Identity")
plt.legend()
plt.grid(True)
plot_file = outfile.replace(".csv", "_plot.png")
plt.savefig(plot_file, dpi=300)
plt.close()

print(f"Saved plot to {plot_file}")



