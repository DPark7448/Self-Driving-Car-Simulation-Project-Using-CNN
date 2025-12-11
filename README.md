Autonomous Driving – Behavioral Cloning
=======================================

Approach
--------
- The model is basically the classic NVIDIA end-to-end driving network. I’m using ELU activations, some L2 regularization and dropout, and the input frames are 66×200 in YUV format.
- For preprocessing, I crop out the sky and car hood, convert to YUV, blur a bit, resize, and normalize. I also optionally clip extreme steering values and rebalance the steering histogram so the model isn’t overwhelmed by straight-road data. Augmentations (flips, small shifts, brightness changes, rotations) plus left/right camera sampling help teach the model how to recover.
- Training uses a train/val split with Adam and MSE/MAE. I added early stopping, learning-rate decay, and model checkpoints. Steering histograms and loss curves get saved into the artifacts/ folder.
- During driving, the script loads the trained model, preprocesses each frame the same way, and then lightly smooths the predicted steering with a median filter + rate limiting. Throttle is automatically reduced when the steering angle gets bigger to keep things stable.

Challenges and mitigations
--------------------------
- The model would constantly predict to drive straight (0 degree turn) so I added steering clipping and an optional histogram rebalance so that my training batches would be less biased.
- Noisy/sparse steering at edges so I added augmentations (flip/translate/rotate) and side-camera sampling with steering correction to help the model learn recovery behavior.
- Since Keras 3 prefers the .keras format, I added a small safeguard to save checkpoints correctly even if .h5 is requested.
- Model struggles to function properly at high speeds so I had manually reduce the throttle to 0.1

Environment setup
-----------------
- Python 
- Create env and install deps:
  ```
  python -m venv .venv
  .venv\Scripts\activate
  pip install --upgrade pip
  pip install -r package_list.txt
  ```
Training
--------
- Basic run with defaults:
  ```
  python src/train.py --data-dir data --output artifacts/model.keras --plots-dir artifacts
  ```
- Helpful flags:
  - `--balance` to enable steering histogram balancing.
  - `--clip-angle 0.9` to clamp extreme labels.
  - `--batch-size`, `--epochs`, `--lr`, `--val-split`, `--l2` to tune training.
- Outputs: best model at `--output`, histograms/training curves in `--plots-dir`.

Driving with simulator
----------------------
- Start server (use the best model path):
  ```
  python src/drive.py --model-path artifacts/model.keras --throttle 0.10 --host 0.0.0.0 --port 4567
  ```
- In the Udacity simulator, select Autonomous Mode. Once the python script is launched the car should start on its own.
- The model does not work very well at higher throttle rates.

Notes
-----
- Models saved in `model.keras`/`model.h5` at the repo root are examples.
- Figures in `artifacts/` are overwritten every training run, save them somewhere else for comparison if desired.
