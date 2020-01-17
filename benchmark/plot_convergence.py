import json

import matplotlib.pyplot as plt
import numpy as np

filename = "../build/davidson.json"
with open(filename, "r") as json_file:
    data = json.load(json_file)


iter_timings = data["davidson"]["iter timing (s)"]
delta_lambda = data["davidson"]["delta lambda"]
subspace_size = data["davidson"]["subspace size"]
iterations = range(len(iter_timings))


fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(delta_lambda)
axs[0, 0].set_ylabel("Eigenvalue Error")

axs[0, 1].plot(subspace_size)
axs[0, 1].set_ylabel("Number of Trial Vectors")

axs[1, 0].plot(iter_timings)
axs[1, 0].set_ylabel("Iteration Time (s)")

# axs[3].plot()
plt.tight_layout()
plt.savefig("davidson_benchmark.png", dpi=600)
# plt.show()

