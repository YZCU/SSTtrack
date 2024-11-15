
---------------- Note ----------------
***Training Stage***
Training set: hotc20 training set
GPU: 1x NVIDIA GeForce RTX 3090
CPU: Intel(R)Xeon(R) Silver 4210 CPU @ 2.20GHz

***Test Stage***
Test Sets: hotc20test, hotc23val_nir, hotc23val_rednir, hotc23val_vis, hotc24val_nir, hotc24val_rednir, hotc24val_vis, mssot, msvt
GPU: NVIDIA GeForce RTX 3090 for hotc20test, NVIDIA GeForce RTX 4060 for others
CPU: Intel(R)Xeon(R) Silver 4210 CPU @ 2.20GHz for hotc20test, 12th Gen Intel (R) Core (TM) i7-12700F for others

- These results have been evaluated using a single model trained on the hotc2020 training set. 
- For the different kinds of hyperspectral data, we only use the first 16 bands in a uniform fashion.
- Tracking performance is expected to be further improved by using corresponding training sets for training.

- The mssot dataset is from 'SFA-guided mosaic transformer for tracking small objects in snapshot spectral imaging'.
- The msvt dataset is from 'Histograms of oriented mosaic gradients for snapshot spectral image description'.
- The other datasets are from 'https://www.hsitracking.com/'.
