import pickle
import numpy as np


errors_file = '../gts_and_errors.pkl'
with open(errors_file, 'rb') as f:
    errors = pickle.load(f)

for filename, data in errors.items():
    breakpoint()
    for i in range(len(data)):
        if data[i]['RE'] > 0:
            print(f"Image #{filename.split('/')[-1]}, i: {i}, TE: {data[i]['TE']}, ")
            breakpoint()