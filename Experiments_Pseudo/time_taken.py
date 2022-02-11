from os import walk
import re
import numpy as np

_, _, filenames = next(walk("./"))

times_taken = np.zeros(shape=(10, 10))

for f_name in filenames:

    if f_name[-3:] != "out":
        continue

    with open("./" + f_name) as f:
        data = f.read()
        der_done = [m.start() for m in re.finditer('Total time taken: ', data)]

        if data.find("Traceback") != -1:
            print("Error in")
            print(f_name)

        for done_idx, done in enumerate(der_done):
            #print(data[done+18:done+30])
            #break
            time_taken = float(data[done + 18:done + 27])
            times_taken[len(der_done)][done_idx] += time_taken
            
            
print(np.sum(times_taken)/3600)