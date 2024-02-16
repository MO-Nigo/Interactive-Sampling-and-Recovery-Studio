# Interactive-Sampling-and-Recovery-Studio
# Sampling-Theory Studio

Sampling-Theory Studio is a desktop application developed to illustrate signal sampling and recovery, showcasing the importance and validation of the Nyquist rate. The application allows users to visualize, sample, and recover signals, as well as compose mixed signals, add noise, and explore different sampling scenarios.

## Features

### Sample & Recover
- Load a mid-length signal and visualize it.
- Sample the signal at different frequencies.
- Recover the original signal using the Whittaker–Shannon interpolation formula.
- Display three graphs: original signal with sampled points, reconstructed signal, and difference between the original and reconstructed signals.

### Load & Compose
- Load signals from files or compose mixed signals within the application.
- Add multiple sinusoidal signals of different frequencies and magnitudes.
- Remove components from the mixed signal.

### Additive Noise
- Add noise to the loaded signal with customizable Signal-to-Noise Ratio (SNR) levels.
- Visualize the effect of noise on the signal frequency.

### Real-time Processing
- Sampling and recovery are performed in real time upon user changes without needing a separate update or refresh button.

### Resize
- The application UI is designed to be easily resizable without affecting functionality.

### Different Sampling Scenarios
- Prepare at least 3 testing synthetic signals that address various sampling scenarios, including tricky features.
- One example is a mix of 2Hz and 6Hz sinusoidals, where sampling at 12Hz or above recovers both frequencies, but sampling at 4Hz may result in aliasing, and sampling at 8Hz may produce unexpected results.

## Getting Started

To run the application locally, follow these steps:

1. Clone the repository to your local machineMM

pip install -r requirements.txt
