# LMTID
Leveraging Multiple Metrics to Improve Difficulty Calibration in Membership Inference Attacks 
# Environment dependencies
- pytorch 2.0.1
- CUDA 12.1
- python 3.9
# Directory structure
+ LMTID
  + data
    + cal_data
      - target_data.pt
      - shadow_data.pt
    + cifar-10-batches-py-official
    + CIFAR10
  + model_IncludingDistillation
    + CIFAR10
      - Shadow
      - Target
  + results 
  + attackMethodsFramework.py
  + LMTID.py
  + Metrics.py
  + MetricsSequence.py
  + Models.py
  + readData.py
+ README.md
# Supported Dataset and Model
_CIFAR10 CIFAR100 CINIC10 SVHN Location<br/>VGG16 MobileNetV2 ResNet50 DenseNet121_
# Usage instructions
We use VGG16 and CIFAR10 as examples to illustrate how to run_LMTID_<br/>
First, please download the CIFAR10 and put into `cifar-10-batches-py-official`.
Second, run He et al.'s method and get the calibrate membership score(save in `target_data.pt`and `shadow_data.pt`) .He et al .'s code can download on [Is Difficulty Calibration All We Need? Towards More Practical Membership Inference Attacks](https://github.com/T0hsakar1n/RAPID)
Once the above two steps are completed,you can run `attackMethodsFramework.py` to start the entire attack process.
# Acknowledgements
Our code is built upon the official repositories of  [SeqMIA: Sequential-Metric Based Membership Inference Attack](https://github.com/AIPAG/SeqMIA) (Li et al., ACM CCS 24) and [Is Difficulty Calibration All We Need? Towards More Practical
 Membership Inference Attacks](https://github.com/T0hsakar1n/RAPID) (He et al., ACM CCS 24). We sincerely appreciate their valuable contributions to the community.
