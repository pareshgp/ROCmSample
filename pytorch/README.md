This folder contains diffrent sample for pytorch
1. profile_pytorch.py - This file contains the sample for profiling pytorch for resnet18 model. This test run with ROCm pytorch docker "rocm/pytorch:latest". 
  To run a sample program execute command "python3 profile_pytorch.py 2>&1 | tee out.log".
  Output for profiling data with ROCm on AMD GPU is available in out.log
