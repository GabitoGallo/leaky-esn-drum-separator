# leaky-esn-drum-separator
A drum source separation model based on Complex Systems theory. Developed during my Physics degree at UNIFI, it utilizes RNNs with edge-of-chaos dynamics and FFT to extract percussion from complex audio mixes.

## Directory Structure
Before running the script (`esn_training_V3.py`), ensure your project root contains the following folders:

* `Dataset/`: Contains the training audio files (e.g., `[name]_mix.wav` and `[name]_drum.wav`).
* `Brani Input/`: Place the `.wav` files you want to separate here.
* `Modelli/`: The script will save the trained `.npz` model here.
* `Tracce Separate/`: The output extracted drum tracks will be saved here.
