# SatelliteAttitudeEstimation
A tool for 3D rotation satellite attitude estimation, is tested on the dataset buaa-sid-pose-1.0.


The method details, dataset description and experimental results can be found in our paper "Deep-Learning-Based Direct Attitude Estimation for Uncooperative Known Space Objects"


This tool provides a framework, researchers can replace their own methods in the framework, you figure out how to use it by yourselves.


This tool is written based on pytorch, it needs to support pytorch3d toolkit, if you need to run and test the bingham loss, please refer to the installation and setup method of torch_bingham library in [DeepBinghamNetwork](https://github.com/Multimodal3DVision/torch_bingham).
