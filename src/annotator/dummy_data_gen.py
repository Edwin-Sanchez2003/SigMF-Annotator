import numpy as np
import json
from pathlib import Path

def create_sigmf_dataset(output_dir="./sigmf_dataset", num_samples=1000000):
    """
    Generate a SigMF dataset with fake CF32_LE IQ data containing periodic chirps.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of complex samples to generate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Parameters
    sample_rate = 40e6  # 40 MHz sample rate (to match 40 MHz bandwidth)
    center_freq = 2.4e9  # 2.4 GHz center frequency
    bandwidth = 40e6  # 40 MHz bandwidth
    
    # Chirp parameters
    chirp_duration = 0.01  # 10 ms chirp duration
    chirp_interval = 0.1  # 100 ms between chirp starts
    chirp_samples = int(chirp_duration * sample_rate)
    interval_samples = int(chirp_interval * sample_rate)
    
    # Generate IQ data
    t = np.arange(num_samples) / sample_rate
    iq_data = np.zeros(num_samples, dtype=np.complex64)
    
    # Add noise background
    iq_data[:] = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.1
    
    # Add periodic chirps
    chirp_start = 0
    while chirp_start + chirp_samples < num_samples:
        chirp_end = chirp_start + chirp_samples
        chirp_t = np.arange(chirp_samples) / sample_rate
        
        # Linear chirp from -bandwidth/2 to +bandwidth/2
        f0 = -bandwidth / 2
        f1 = bandwidth / 2
        chirp_rate = (f1 - f0) / chirp_duration
        phase = 2 * np.pi * (f0 * chirp_t + 0.5 * chirp_rate * chirp_t**2)
        
        # Add chirp with amplitude 0.5
        iq_data[chirp_start:chirp_end] += 0.5 * np.exp(1j * phase).astype(np.complex64)
        
        chirp_start += interval_samples
    
    # Save the data file
    data_file = output_dir / "data.sigmf-data"
    iq_data.tofile(data_file)
    
    # Create metadata file
    metadata = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": sample_rate,
            "core:hw_info": "Fake IQ generator with chirps",
            "core:author": "SigMF Generator",
            "core:description": "Randomly generated IQ data with periodic chirps spanning 40 MHz",
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": center_freq,
                "core:datetime": "2024-01-01T00:00:00Z",
            }
        ],
        "annotations": [],
    }
    
    # Save metadata file
    metadata_file = output_dir / "data.sigmf-meta"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"SigMF dataset created in {output_dir}")
    print(f"  Data file: {data_file} ({iq_data.nbytes} bytes)")
    print(f"  Metadata file: {metadata_file}")
    print(f"  Samples: {num_samples}")
    print(f"  Sample rate: {sample_rate/1e6} MHz")
    print(f"  Center frequency: {center_freq/1e9} GHz")
    print(f"  Bandwidth: {bandwidth/1e6} MHz")
    print(f"  Chirp duration: {chirp_duration*1000} ms")
    print(f"  Chirp interval: {chirp_interval*1000} ms")

if __name__ == "__main__":
    create_sigmf_dataset(num_samples=500_000_000)
