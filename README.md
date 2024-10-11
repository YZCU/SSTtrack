# SSTtrack
The official implementation for [SSTtrack: A Unified Hyperspectral Video Tracking Framework via Modeling Spectral-Spatial-Temporal Conditions](https://www.sciencedirect.com/science/article/pii/S1566253524004366)

--------------------------------------------------------------------------------------
:running:Keep updating:running:: More detailed tracking results for SSTtrack have been released.
- [hotc20test](https://www.hsitracking.com/) ([results](https://github.com/YZCU/SSTtrack/tree/master/tracking_results))
- [hotc23val_nir](https://www.hsitracking.com/) ([results](https://github.com/YZCU/SSTtrack/tree/master/tracking_results))
- [hotc23val_rednir](https://www.hsitracking.com/) ([results](https://github.com/YZCU/SSTtrack/tree/master/tracking_results))
- [hotc23val_vis](https://www.hsitracking.com/) ([results](https://github.com/YZCU/SSTtrack/tree/master/tracking_results))
- [hotc24val_nir](https://www.hsitracking.com/) ([results](https://github.com/YZCU/SSTtrack/tree/master/tracking_results))
- [hotc24val_rednir](https://www.hsitracking.com/) ([results](https://github.com/YZCU/SSTtrack/tree/master/tracking_results))
- [hotc24val_vis](https://www.hsitracking.com/) ([results](https://github.com/YZCU/SSTtrack/tree/master/tracking_results))
- [mssot](https://www.sciencedirect.com/science/article/pii/S0924271623002551) ([results](https://github.com/YZCU/SSTtrack/tree/master/tracking_results))
- [msvt](https://www.sciencedirect.com/science/article/pii/S0924271621002860) ([results](https://github.com/YZCU/SSTtrack/tree/master/tracking_results))
--------------------------------------------------------------------------------------


##  Install
```
git clone https://github.com/YZCU/SSTtrack.git
```
## Environment
 > * CUDA 11.8
 > * Python 3.9.18
 > * PyTorch 2.0.0
 > * Torchvision 0.15.0
 > * numpy 1.25.0 
 - Please check the `requirement.txt` for details.

## Usage
- Download the RGB/Hyperspectral training/test datasets: [LaSOT](https://cis.temple.edu/lasot/), [GOT-10K](http://got-10k.aitestunion.com/downloads), [COCO](http://cocodataset.org), [HOTC](https://www.hsitracking.com/hot2022/), and [TrackingNet](https://tracking-net.org/#downloads).
- Download the pretrained model: [pretrained model](https://pan.baidu.com/s/19pmFUAA0Bvj0s0GP_4xccA) (code: abcd) to `pretrained_models/`.
- Please train the SSTtrack based on the [foundation model](https://pan.baidu.com/s/19pmFUAA0Bvj0s0GP_4xccA) (code: abcd).
- We will release the well-trained model of [SSTtrack](https://pan.baidu.com/s/1mNefFkZ1Z6c0PRCZVHYjEgg) (code: abcd).
- The generated model will be saved to the path of `output/train/ssttrack/ssttrack-ep150-full-256/`.
- Please test the model. The results will be saved in the path of `output/results/ssttrack/ssttrack-ep150-full-256/otb`.
- For evaluation, please download the evaluation benchmark [Toolkit](http://cvlab.hanyang.ac.kr/tracker_benchmark/) and [vlfeat](http://www.vlfeat.org/index.html) for more precision performance evaluation.
- Refer to [HOTC](https://www.hsitracking.com/hot2022/) for evaluation.
- Evaluation of the SSTtrack tracker. Run `\tracker_benchmark_v1.0\perfPlot.m`
- Relevant tracking results are provided in `SSTtrack\tracking_results\hotc20test`. More evaluation results are provided in a `SSTtrack\tracking_results`.

## Results
- Multi-modal samples generated from the hyperspectral modality
 ![image](/fig/00.gif)
 ![image](/fig/11.gif)
 ![image](/fig/22.gif)
 ![image](/fig/33.gif)

- Qualitative results
 ![image](/fig/12.jpg)
 ![image](/fig/v1.gif)
 ![image](/fig/v2.gif)
 ![image](/fig/v3.gif)

:heart:  :heart:

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: yuzeng_chen@whu.edu.cn 
 
## Citation
If you find our work helpful in your research, kindly consider citing it. We appreciate your support.


@article{CHEN2025102658,
title = {SSTtrack: A unified hyperspectral video tracking framework via modeling spectral-spatial-temporal conditions},
journal = {Information Fusion},
volume = {114},
pages = {102658},
year = {2025},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2024.102658},
url = {https://www.sciencedirect.com/science/article/pii/S1566253524004366},
author = {Yuzeng Chen and Qiangqiang Yuan and Yuqi Tang and Yi Xiao and Jiang He and Te Han and Zhenqi Liu and Liangpei Zhang},
keywords = {Hyperspectral video, Spectral awareness, Temporal awareness, Prompt learning, Multi-modal tracking},
abstract = {Hyperspectral video contains rich spectral, spatial, and temporal conditions that are crucial for capturing complex object variations and overcoming the inherent limitations (e.g., multi-device imaging, modality alignment, and finite spectral bands) of regular RGB and multi-modal video tracking. However, existing hyperspectral tracking methods frequently encounter issues including data anxiety, band gap, huge volume, and weakness of the temporal condition embedded in video sequences, which result in unsatisfactory tracking capabilities. To tackle the dilemmas, we present a unified hyperspectral video tracking framework via modeling spectral-spatial-temporal conditions end-to-end, dubbed SSTtrack. First, we design the multi-modal generation adapter (MGA) to explore the interpretability benefits of combining physical and machine models for learning the multi-modal generation and bridging the band gap. To dynamically transfer and interact with multiple modalities, we then construct a novel spectral-spatial adapter (SSA). Finally, we design a temporal condition adapter (TCA) for injecting the temporal condition to guide spectral and spatial feature representations to capture static and instantaneous object properties. SSTtrack follows the prompt learning paradigm with the addition of few trainable parameters (0.575 M), resulting in superior performance in extensive comparisons. The code will be released at https://github.com/YZCU/SSTtrack.}
}
