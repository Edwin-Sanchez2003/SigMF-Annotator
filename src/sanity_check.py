import sigmf
import numpy as np

handle = sigmf.sigmffile.fromfile("/var/home/edwsanch/Downloads/trimmedSamples.sigmf-data")
samples = handle.read_samples(start_index=0, count=100)
print(samples[:5])
