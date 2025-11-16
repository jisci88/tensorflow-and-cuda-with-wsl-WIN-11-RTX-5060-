# tensorflow-and-cuda-with-wsl-WIN-11-RTX-5060-


RTX50시리즈는 tensorflow가 정식으로 지원하지는 않는다고 함. 나는 RTX5060 유저라 wsl 환경을 구축하면서 많은 실패를 했는데, 결국 성공적으로 구성할 수 있어 CUDA 버전과 여러 세팅들을 정리하고 함.



Anaconda 설치 방법
https://datanavigator.tistory.com/62


RTX 50 시리즈 기준 tensorflow 설치
https://www.philgineer.com/2025/10/50605090-gpu-pytorch-tensorflow.html

=> 요약.
 wsl ubuntu 22.04
 cuda 12.8 - https://developer.nvidia.com/cuda-12-8-0-download-archive (deb local에서 설치 가이드 그대로 수행), sudo apt-get install python3-pip, pip3 install nvidia-cudnn-cu12
 python 3.10
 torch 2.8.0+cu128 (Pytorch 사용 시)
 torchvision 0.23.0+cu128 (//)
 tf_nightly 2.21.0.dev20250920 ( tensorflow 사용 시 pip install tf-nightly[and-cuda])
 (출처: 위 블로그)

vim ~/.bashrc 에 
export PATH=$PATH:/usr/local/cuda-12.8/bin
export CUDADIR=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.8/lib64
추가



우분투 한글화
https://datanavigator.tistory.com/60
