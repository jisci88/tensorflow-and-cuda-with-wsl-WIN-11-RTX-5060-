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



! 설치하면서 유용한 명령어들


! gpu 연동 확인
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))



! 우분투 삭제
wslconfig /u Ubuntu-22.04



! 버전
import tensorflow as tf
print(tf.version.VERSION)      # TensorFlow 버전
print(tf.sysconfig.get_build_info()['cuda_version'])  # 빌드된 CUDA 버전
print(tf.sysconfig.get_build_info()['cudnn_version']) # 빌드된 cuDNN 버전



! goolge chrome 설치
cd /tmp
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install --fix-broken -y ./google-chrome-stable_current_amd64.deb



! 커널 등록하기
conda install ipykernel -y
python -m ipykernel install --user --name=tf-gpu --display-name "Python (tf-gpu)"



! 등록된 커널 삭제
jupyter kernelspec list
jupyter kernelspec uninstall tf-gpu




! 테스트 간단 학습
import tensorflow as tf
import numpy as np


x_train = np.array([0, 1, 2, 3, 4, 5], dtype=float)
y_train = np.array([3, 5, 7, 9, 11, 13], dtype=float)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])


model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=500, verbose=0)  # verbose=0 -> 학습 진행 표시 안함

x_test = np.array([6, 7, 8], dtype=float)
y_pred = model.predict(x_test)

for i, x in enumerate(x_test):
    print(f"x={x} -> y_pred={y_pred[i][0]:.2f}")
