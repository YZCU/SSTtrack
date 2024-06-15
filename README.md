# SSTtrack

The official implementation for ["**SSTtrack: A Unified Hyperspectral Video Tracking Framework via Modeling Spectral-Spatial-Temporal Conditions**"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4860918)

- Authors: 
[Yuzeng Chen](https://yzcu.github.io/), 
[Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/),
[Yuqi Tang](https://faculty.csu.edu.cn/yqtang/zh_CN/zdylm/66781/list/index.htm),
[Yi Xiao](https://xy-boy.github.io/),
[Jiang He](https://jianghe96.github.io/),
[Te Han](https://jianghe96.github.io/),
[Zhenqi Liu](http://ai.swu.edu.cn/info/1067/2670.htm), and 
[Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)

- Wuhan University ([School of Geodesy and Geomatics](http://main.sgg.whu.edu.cn/), [State Key Laboratory of Information Engineering, Survey Mapping and Remote Sensing](https://liesmars.whu.edu.cn/info/1058/16203.htm)), Central South University ([School of Geosciences and Info-Physics](https://gip.csu.edu.cn/index.htm), Southwest University ([College of Artificial Intelligence](http://ai.swu.edu.cn/index.htm), and

## Abstract
> Hyperspectral video contains rich spectral, spatial, and temporal conditions that are crucial for capturing complex object variations and overcoming the inherent limitations (e.g., multi-device imaging, modality alignment, and finite spectral bands) of regular RGB and multi-modal video tracking. However, existing hyperspectral tracking methods frequently encounter issues including data anxiety, band gap, huge volume, and weakness of the temporal condition embedded in video sequences, which result in unsatisfactory tracking performance. To address these dilemmas, we propose a unified hyperspectral video tracking framework via modeling spectral-spatial-temporal conditions in an end-to-end fashion, dubbed SSTtrack. First, we design a multi-modal generation adapter (MGA) to explore the interpretability benefits of combining physical and machine models for learning the multi-modal generation and bridging the band gap. Then, we construct a spectral-spatial adapter (SSA) to dynamically transfer and interact with multiple modalities. Finally, we design a temporal condition adapter (TCA) for injecting the temporal condition to guide spectral and spatial feature representations to capture static and instantaneous object properties. SSTtrack follows the prompt learning paradigm with the addition of fewer trainable parameters (0.575M), resulting in superior performance in extensive comparisons. The code and model will be available at https://github.com/YZCU/SSTtrack.
> 
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
 - **Note:** Please check the `requirement.txt` for details.
## Download datasets
- **RGB/Hyperspectral training/test datasets:**
 > * [GOT-10K](http://got-10k.aitestunion.com/downloads), 
 > [DET](http://image-net.org/challenges/LSVRC/2017/), 
 > [LaSOT](https://cis.temple.edu/lasot/),
 > [COCO](http://cocodataset.org),
 > [YOUTUBEBB](https://pan.baidu.com/s/1gQKmi7o7HCw954JriLXYvg) (code: v7s6),
 > [VID](http://image-net.org/challenges/LSVRC/2017/),
 > [HOTC](https://www.hsitracking.com/hot2022/)
- **Note:** Please download these datasets and Codes.

## Usage
### Quick Start
- **Step I.**  Download the RGB/Hyperspectral training/test datasets:
[GOT-10K](http://got-10k.aitestunion.com/downloads), 
[DET](http://image-net.org/challenges/LSVRC/2017/), 
[LaSOT](https://cis.temple.edu/lasot/),
[COCO](http://cocodataset.org),
[YOUTUBEBB](https://pan.baidu.com/s/1gQKmi7o7HCw954JriLXYvg) (code: v7s6),
[VID](http://image-net.org/challenges/LSVRC/2017/),
[HOTC](https://www.hsitracking.com/hot2022/),
and put them in the path of `train_dataset/dataset_name/`.
- **Step II.**  Download the pretrained model: [pretrained model](https://pan.baidu.com/s/1ZW61I7tCe2KTaTwWzaxy0w) to `pretrained_models/`.
- **Step III.**  Run the `setup.py` to set the path.
- **Step IV.**  To train a model, switch directory to `tools/` and run `train.py` with the desired configs.
### Only Test
- **Step I.**  We will release the trained [SSTtrack model](https://). Please put it to the path of `tools/snapshot/`.
- **Step II.**  Switch directory to `tools/` and run the `tools/test.py`.
- **Step III.**  Results will be saved in the path of `tools/results/`.
### Evaluation
- **Step I.**  Please download the evaluation benchmark [Toolkit](http://cvlab.hanyang.ac.kr/tracker_benchmark/) and [vlfeat](http://www.vlfeat.org/index.html) for more precision performance evaluation.
- **Step II.**  Download the file of the `tracking results` and put it into the path of `\tracker_benchmark_v1.0\results\results_OPE_SSTtrack`.
- **Step III.**  Evaluation of the SSTtrack tracker. Run `\tracker_benchmark_v1.0\perfPlot.m`

## Results
- Multi-modal samples generated from the hyperspectral modality
 ![image](/fig/00.gif)
 ![image](/fig/11.gif)
 ![image](/fig/22.gif)
 ![image](/fig/33.gif)

- Comparison with SOTA hyperspectral trackers
 ![image](/fig/5.jpg)
- *Hyperspectral videos: (a) Precision plot. (b) Success plot*
 ![image](/fig/6.jpg)
- *Accuracy-speed comparisons. (a) Pre vs. FPS. (b) Suc vs. FPS*
 ![image](/fig/7.jpg)
 
- Comparison with SOTA RGB trackers
 ![image](/fig/0.jpg)
 
- Comparison with hand-crafted feature-based trackers
- *RGB videos: (a) Precision plot. (b) Success plot*
 ![image](/fig/1.jpg)
- *False color videos: (a) Precision plot. (b) Success plot*
 ![image](/fig/2.jpg)
 
- Comparison with deep feature-based trackers
- *RGB videos: (a) Precision plot. (b) Success plot*
 ![image](/fig/3.jpg)
- *False color videos: (a) Precision plot. (b) Success plot*
 ![image](/fig/4.jpg)
 
- Attribute-based Evaluations
- *Pre results for each attribute and overall*
 ![image](/fig/8.jpg)
- *Suc results for each attribute and overall*
 ![image](/fig/9.jpg)

- *Precision plots for each attribute and overall*
 ![image](/fig/10.jpg)
- *Success plots for each attribute and overall*
 ![image](/fig/11.jpg)

- Qualitative results
 ![image](/fig/12.jpg)
  ![image](/fig/v1.gif)
   ![image](/fig/v2.gif)
   ![image](/fig/v3.gif)
:heart:For more comprehensive results, please review the upcoming manuscript:heart:

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: yuzeng_chen@whu.edu.cn 
 
## Citation
If you find our work helpful in your research, kindly consider citing it. We appreciate your support.


```
@ARTICLE{,
  author={},
  journal={}, 
  title={}, 
  year={},
  volume={},
  number={},
  pages={},
  keywords={},
  doi={}
```

