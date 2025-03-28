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
First, please download the dataset and put into the correct dictoraty.
Second, run He et al.'s method and get the calibrate membership score(save in `target_data.pt`and `shadow_data.pt`) .He et al .'s code can download on [Is Difficulty Calibration All We Need?](https://github.com/T0hsakar1n/RAPID))

Then, you can run attackMethodsFramework.py to start the entire attack process.
