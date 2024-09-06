# Main1 data processing

# Set random seed 
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# List all stations and their epicentral distances in the HDF5 file
def list_stations_in_file(file_path):
    """List all stations and their epicentral distances in the HDF5 file"""
    stations_info = {}
    with h5py.File(file_path, 'r') as f:
        for station in f.keys():
            distance = f[station].attrs['dist_m']
            stations_info[station] = distance
    return stations_info

# High-pass filter function to filter out frequency components below the cutoff frequency
def highpass_filter(data, cutoff=1, fs=100, order=4):
    """Apply high-pass filter to data"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# Low-pass filter function to filter out frequency components above the cutoff frequency
def lowpass_filter(data, cutoff=20.0, fs=100, order=4):
    """Apply low-pass filter to data"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Calculate P-wave arrival time based on epicentral distance
def calculate_p_wave_time(distance_m, vp=6000.0):
    """Calculate P-wave arrival time based on epicentral distance"""
    distance_m = float(distance_m)  # Convert distance to float
    return distance_m / vp

# Load and preprocess data function
def load_and_preprocess_data(file_path, station, distance, num_channels=3, fs=100):
    """Load and preprocess data, including high-pass and low-pass filtering, demeaning, and normalization"""
    with h5py.File(file_path, 'r') as f:
        if station not in f:
            raise KeyError(f"Station {station} not found in file {file_path}")
        data = f[station][0:num_channels, :]

    # Calculate P-wave arrival time
    p_wave_time = calculate_p_wave_time(distance)
    print(f"P-wave arrival time at station {station}: {p_wave_time} seconds")

    # Calculate the start and end indices for data slicing
    start_index = max(0, int((p_wave_time - 10) * fs))
    end_index = min(start_index + 600 * fs, data.shape[1])

    # Slice data from 10 seconds before to 600 seconds after P-wave arrival
    p_wave_data = data[:, start_index:end_index]
    print(f"Data shape after slicing at station {station}: {p_wave_data.shape}")

    # Apply high-pass and low-pass filtering to the data
    filtered_data = np.array([lowpass_filter(highpass_filter(p_wave_data[i, :], cutoff=1, fs=fs), cutoff=20.0, fs=fs) for i in range(num_channels)])

    # Demean and normalize data
    detrended_data = filtered_data - np.mean(filtered_data, axis=1, keepdims=True)
    normalized_data = detrended_data / np.std(detrended_data, axis=1, keepdims=True)

    # Print the shape of the normalized time-domain data
    print(f"Shape of normalized time-domain data: {normalized_data.shape}")
    
    return normalized_data

# Short-Time Fourier Transform (STFT) function
def compute_stft(normalized_data, fs=100, nperseg=256, noverlap=192):
    """Compute STFT for each channel and normalize"""
    Zxx_log_normalized = [None] * normalized_data.shape[0]
    for i in range(normalized_data.shape[0]):
        _, _, Zxx = stft(normalized_data[i], fs=fs, window='hamming', nperseg=nperseg, noverlap=noverlap)
        Zxx_log = np.log(np.abs(Zxx) + 1e-10)
        Zxx_log_normalized[i] = (Zxx_log - np.mean(Zxx_log, axis=(0, 1))) / np.std(Zxx_log, axis=(0, 1))
    
    # Stack data from different channels into a matrix of shape (251, 101, 3)
    Zxx_log_normalized_stacked = np.stack(Zxx_log_normalized, axis=-1)
    print(f"STFT computation complete: Zxx_log_normalized_stacked shape: {Zxx_log_normalized_stacked.shape}")
    return Zxx_log_normalized_stacked

# Process all event files and extract features (batch processing)
def process_all_events_in_batches(base_dir, event_files, fs=100, batch_size=5):
    """Process all event files and extract frequency features"""
    all_freq_features_combined = []

    for batch_start in range(0, len(event_files), batch_size):
        batch_files = event_files[batch_start:batch_start + batch_size]
        freq_batch = []

        for event_file in batch_files:
            file_path = os.path.join(base_dir, f"{event_file}.h5")
            stations = list_stations_in_file(file_path)

            for station, distance in stations.items():
                try:
                    normalized_data = load_and_preprocess_data(file_path, station, distance, fs=fs)
                    Zxx_log_normalized = compute_stft(normalized_data, fs=fs)
                    freq_batch.append(Zxx_log_normalized)
                except KeyError as e:
                    print(f"Error processing {event_file} - {station}: {e}")

        if freq_batch:
            # Use np.concatenate to preserve the channel dimension, combining frequency feature data
            freq_batch_combined = np.concatenate(freq_batch, axis=0)
            print(f"Batch {batch_start}: Combined frequency data shape: {freq_batch_combined.shape}")
            all_freq_features_combined.append(freq_batch_combined)
            np.save(os.path.join(base_dir, f"freq_features_batch_{batch_start}.npy"), freq_batch_combined)

    if all_freq_features_combined:
        all_freq_features_combined = np.concatenate(all_freq_features_combined, axis=0)
    else:
        all_freq_features_combined = np.array([])

    print(f"Shape of all combined frequency features: {all_freq_features_combined.shape}")
    return all_freq_features_combined

# Main program 1: Responsible for loading and processing data, and extracting frequency features
def main1():
    base_dir = "E:\japan"
    event_files = ["file name"]
    fs = 100  # Sampling frequency
    batch_size = 10 
    
    print("Processing all event files in batches...")
    freq_features = process_all_events_in_batches(base_dir, event_files, fs=fs, batch_size=batch_size)
    
    if freq_features.size == 0:
        print("No frequency features extracted, exiting program...")
        return
    
    # Save features to disk for use in main2
    np.save(os.path.join(base_dir, "freq_features.npy"), freq_features)
    print(f"Frequency features saved to: {os.path.join(base_dir, 'freq_features.npy')}")

if __name__ == "__main__":
    main1()
