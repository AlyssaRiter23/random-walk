import os
import csv
import math
import time
import pickle
import glob
import numpy as np
import threading
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import tqdm
import concurrent.futures
import multiprocessing

# Usage: face_ear_detector glob outfile.csv

if len(sys.argv) < 3:
    print(f'Usage: {sys.argv[0]} glob outfile.csv')
    sys.exit(1)

#
# enumerate GPUs and workers - these can be tuned for available GPUs and memory usage.
# There are diminshing returns here as each worker is I/O bound (reading the image). Putting the images
# in a ramdisk might help?
#

GPUS = [0]#,1]
WORKERS = [0] #1] # , 2, 3] #, 4, 5] #, 6, 7]

if len(sys.argv) < 3:
    print(f'Usage: {sys.argv[0]} glob outfile.csv')
    sys.exit(1)

globpattern = sys.argv[1]
outfile = sys.argv[2]

filenames = glob.glob(sys.argv[1])
if len(filenames) == 0:
    print(f"Glob {glob} didn't match any files.")
    sys.exit(1)

print(f'{len(filenames)} files to examine.')

local = threading.local()

def initializer(q,local):
    [local.g,local.w] = q.get()
    #print(f'Initializing GPU {local.g}, worker {local.w}')
    provider = [('CUDAExecutionProvider',{'device_id':local.g})]
    local.app = FaceAnalysis(providers=provider, allowed_modules=['detection', 'recognition'])
    local.app.prepare(ctx_id=local.w)

def worker(fn,local):
    print(f"Running GPU {local.g}'s worker {local.w} on file {fn}")
    local.fn = fn
    local.vc = cv2.VideoCapture(fn)
    local.res = []
    local.fno = 0
    while True:
        local.ok,local.img = local.vc.read()
        if not local.ok:
            break
        else:
            local.dets = local.app.get(local.img)
            for det in local.dets:
                if "embedding" not in det or det["embedding"] is None:
                    det["embedding"] = np.zeros((512,))  # replace with a zero vector if missing

            #local.res.append({'name': os.path.basename(local.fn), 'frame_number': local.fno, 'results': local.dets})
            #local.fno += 1
            #print(f"Results of running GPU {local.g}'s worker {local.w} on file {fn}, frame {fno}: {dets}")
            local.res.append({'name':os.path.basename(local.fn), 'frame_number':local.fno, 'results':local.dets})
            local.fno += 1

            #for i,d in enumerate(local.dets):
                #l,t,r,b = [int(x) for x in d['bbox']]
                #print(f'{fn}, {[l,t,r,b]}')
                #chip = local.img[t:b,l:r]
                #ts = local.fn.split(os.sep)
                #cv2.imwrite(f'/tmp/stuff/{ts[-2]+'-'+ts[-1]}.chip{i}.png',chip)

    print(f"Finished running GPU {local.g}'s worker {local.w} on file {local.fn}, {len(local.res)} frames")
    return local.res

# whee! we have to serialize the writes using a mutex.

lock = threading.Lock()

def done_callback(f):
    global fp,lock
    results = f.result()
    for fresults in results: # results for each frame, if a video (singleton if image)
        fname = fresults['name']
        fnumber = fresults['frame_number']
        r2 = fresults['results']
        if len(r2) == 0:
            continue
        for i,res in enumerate(r2):
            embedding = res.get("embedding", None)
            embedding_str = f'"{embedding.tolist()}"' if embedding is not None else '"[]"'
            with lock: # serialize
                # have to quote the lists because they contain commas
                #print(f'{fname},{fnumber},{i},\"{((res["bbox"].tolist()))}\",{res["det_score"]},\"{res["kps"].tolist()}\"',file=fp,flush=True)
                print(f'{fname},{res["det_score"]}{embedding_str}', file=fp, flush=True)

futures = []

# initialize workers with a (gpu_id, worker_id) pair

q = multiprocessing.Queue()
for g in GPUS:
    for w in WORKERS:
        q.put([g,w])

# output csv layout
#/data2/flynn/face-data/FRGC/FRGC-2.0-dist/nd1/Spring2004/02463d646.jpg,1: {'bbox': array([183.29956, 701.2343 , 266.6425 , 800.24567], dtype=float32), 'kps': array([[202.62097, 742.04987], [237.2692 , 736.62933], [219.92476, 760.5047 ], [213.39253, 778.03046], [239.60027, 773.64777]], dtype=float32), 'det_score': 0.7131239}

fp = open(outfile,'w')
#print('Filename,frame,index,bbox_ltrb,score,landmarks',file=fp)
print('Filename,score,embedding', file=fp)

with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)*len(WORKERS),initializer=initializer,initargs=(q,local)) as p:
    for ifn,fn in enumerate(tqdm.tqdm(filenames)):
        f = p.submit(worker,fn,local)
        f.add_done_callback(done_callback)

    while len(futures) > 0:
        print(f'Running futures: length is {len(futures)}')
        f = futures.pop(0)
        if not f.done():
            futures.append(f)

fp.close()
