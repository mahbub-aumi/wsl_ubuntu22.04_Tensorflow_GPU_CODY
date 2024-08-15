# wsl_ubuntu22.04_Tensorflow_GPU_CODY
Tensorflow GPU for Nvidia with code assistance for VSCODE

#### System Update and Upgrade
```
sudo apt update && sudo apt upgrade -y
```

#### Install required packages
```
sudo apt install build-essential -y
```

#### Check the CUDA requirements for Tensorflow
```
https://www.tensorflow.org/install/source#gpu
```

#### Download Cuda Toolkit 12.3
```
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
```

#### Install Cuda Toolkit
```
sudo sh cuda_12.3.2_545.23.08_linux.run
```

#### Download cuDNN 8.9 (Local Installer for Linux x86_64 tar)
```
https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz/
```
	You need nvidia developer id to download cuDNN file

#### Extract cuDNN tar file
```
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
```

#### Copy cuDNN files
```
cd cudnn-linux-x86_64-8.9.7.29_cuda12-archive
```
```
sudo cp include/cudnn*.h /usr/local/cuda-12.3/include
```
```
sudo cp lib/libcudnn* /usr/local/cuda-12.3/lib64
```
```
sudo chmod a+r /usr/local/cuda-12.3/include/cudnn*.h /usr/local/cuda-12.3/lib64/libcudnn*
```

#### Verify cuDNN installation
```
cat /usr/local/cuda-12.3/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

#### Download TensorRT 8.6
```
https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
```
		You need nvidia developer id to download this file

#### Extract and Move TensorRT files\
```
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
```
```
sudo mv TensorRT-8.6.1.6 /usr/local/TensorRT-8.6.1
```

#### Modify .bashrc file for Cuda Toolkit, cuDNN and TensorRT
```
nano ~/.bashrc
```
```
export PATH=/usr/local/cuda-12.3/bin:/usr/local/TensorRT-8.6.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:/usr/local/TensorRT-8.6.1/lib:$LD_LIBRARY_PATH
export CUDNN_PATH=/usr/local/cuda-12.3/lib64
```
```
source ~/.bashrc
```


#### Install TensorFlow 2.16.1
```
python -m pip install tensorflow[and-cuda]==2.16.1
```

```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```


```
conda install ipykernel
```

```
python -m ipykernel install --user --name=tf217 --display-name "tf217"
```

```
python -m ipykernel install --user --name=tf216 --display-name "tf216"
```

