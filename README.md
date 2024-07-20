# [SSTtrack](/SSTtrack.pdf)

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

- Wuhan University ([School of Geodesy and Geomatics](http://main.sgg.whu.edu.cn/), [State Key Laboratory of Information Engineering, Survey Mapping and Remote Sensing](https://liesmars.whu.edu.cn/index.htm)), Central South University ([School of Geosciences and Info-Physics](https://gip.csu.edu.cn/index.htm), Southwest University ([College of Artificial Intelligence](http://ai.swu.edu.cn/index.htm)).

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
[LaSOT](https://cis.temple.edu/lasot/),
[GOT-10K](http://got-10k.aitestunion.com/downloads),
[COCO](http://cocodataset.org),
[HOTC](https://www.hsitracking.com/hot2022/),
[TrackingNet](https://tracking-net.org/#downloads),

- Download the pretrained model: [pretrained model](https://pan.baidu.com/s/19pmFUAA0Bvj0s0GP_4xccA), (code:abcd) to `pretrained_models/`.
### Only Test
- **Step I.**  We will release the trained [SSTtrack model](https://). Please put it to the path of `tools/snapshot/`.
- **Step II.**  Switch directory to `tools/` and run the `tools/test.py`.
- **Step III.**  Results will be saved in the path of `tools/results/`.
### Evaluation
- **Step I.**  Please download the evaluation benchmark [Toolkit](http://cvlab.hanyang.ac.kr/tracker_benchmark/) and [vlfeat](http://www.vlfeat.org/index.html) for more precision performance evaluation.
- **Step II.**  Refer to [HOTC](https://www.hsitracking.com/hot2022/) for evaluation.
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
:heart:For more comprehensive results and experimental details, please review the upcoming [Manuscript](/SSTtrack.pdf) and Codes :heart:

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: yuzeng_chen@whu.edu.cn 
 
## Citation
If you find our work helpful in your research, kindly consider citing it. We appreciate your support.

Chen, Yuzeng and Yuan, Qiangqiang and Tang, Yuqi and Xiao, Yi and He, Jiang and Han, Te and Liu, Zhenqi and Zhang, Liangpei, Ssttrack: A Unified Hyperspectral Video Tracking Framework Via Modeling Spectral-Spatial-Temporal Conditions. Available at SSRN: https://ssrn.com/abstract=4860918 or http://dx.doi.org/10.2139/ssrn.4860918

