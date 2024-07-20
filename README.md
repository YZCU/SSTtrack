# [SSTtrack](/SSTtrack.pdf)
The official implementation for ["**SSTtrack: A Unified Hyperspectral Video Tracking Framework via Modeling Spectral-Spatial-Temporal Conditions**"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4860918)
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
- Download the pretrained model: [pretrained model](https://pan.baidu.com/s/19pmFUAA0Bvj0s0GP_4xccA), (code: abcd) to `pretrained_models/`.
- Please train the SSTtrack based on the [foundation model](https://pan.baidu.com/s/19pmFUAA0Bvj0s0GP_4xccA), (code: abcd).
- We will release the well-trained model of [SSTtrack](https://pan.baidu.com/s/19pmFUAA0Bvj0s0GP_4xccA), (code: abcd).
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
:heart:For more comprehensive results and experimental details, please review the upcoming [Manuscript](/SSTtrack.pdf) and Codes :heart:

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: yuzeng_chen@whu.edu.cn 
 
## Citation
If you find our work helpful in your research, kindly consider citing it. We appreciate your support.

Chen, Yuzeng and Yuan, Qiangqiang and Tang, Yuqi and Xiao, Yi and He, Jiang and Han, Te and Liu, Zhenqi and Zhang, Liangpei, Ssttrack: A Unified Hyperspectral Video Tracking Framework Via Modeling Spectral-Spatial-Temporal Conditions. Available at SSRN: https://ssrn.com/abstract=4860918 or http://dx.doi.org/10.2139/ssrn.4860918

