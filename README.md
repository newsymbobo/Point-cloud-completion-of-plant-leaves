# Point-cloud-completion-of-plant-leaves

This code is adapted from the PF-net source code, which is the Pytorch implement of CVPR2020 PF-Net: Point Fractal Network for 3D Point Cloud Completion. 

The source code repository and the address of the paper are listed below:

https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_PF-Net_Point_Fractal_Network_for_3D_Point_Cloud_Completion_CVPR_2020_paper.pdf

https://github.com/zztianzz/PF-Net-Point-Fractal-Network


##0) Environment
Windows
CUDA 11.4 
Python 3.7 
PyTorch 1.9
#NVIDIA GeForce RTX 3060 

also in：
Windows
CUDA 11.2
Python 3.8 
PyTorch 1.12
#NVIDIA GeForce RTX 3060 

##1) Dataset
  You can download the dataset from 
  https://drive.google.com/file/d/1ZRVhjRzJM-vYYZ5LqCHhtddkR_y4FOc4/view?usp=drive_link
  or https://pan.baidu.com/s/1q1r7IUawHZUu94t8FISmbQ
  Replace the 'dataset/resample' folder with the downloaded data.
  Please provide your basic personal information (organization/school and your name). 
  You can also reach me via email at 494251855@qq.com.

##2) Train
```
python train.py 
```

##3) Evaluate the Performance on example
```
python predict.py
```
We provide a example of training results in 'Checkpoint' folder as well as the data used for testing in 'test_example' folder and 'test_real' folder.
The test data in 'test_example' folder come from dataset-test.
The test data in 'test_real' folder captured by Azure-Kinect under natural occlusion.

change ‘--infile’ to select different incomplete point cloud.
Show the completion results, the program will generate txt files in 'fake' folder.

##4) Visualization of Examples

Using Meshlab/cloudcompare to visualize the txt files.
