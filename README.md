# Gaussian Heritage: 3D Digitization of Cultural Heritage with Integrated Object Segmentation
ECCV 2024 VISART Workshop
[Project Page](https://mahtaabdn.github.io/Gaussian-Heritage/#) | [Paper](https://mahtaabdn.github.io/Gaussian-Heritage/#) 
![Preview Image](https://github.com/mahtaabdn/Gaussian-Heritage/blob/gh-pages/fig1.png)
## Overview

This repository contains the implementation of "Gaussian Heritage", a pipeline for 3D digitization of cultural heritage objects using only RGB images. Our method leverages advancements in novel view synthesis and Gaussian Splatting to create 3D replicas of scenes and extract models for individual items of interest.


## Key Features

- Generate 3D replicas of scenes using only multi-view RGB images (e.g., photos from a museum)
- Extract individual 3D models for items of interest within the scene
- Deployable as a Docker container for easy setup and use


## Installation

### Requirements

- Host machine with at least one NVIDIA GPU/CUDA support and installed drivers
- Docker
- Ubuntu 22.04

### Installation

1. Check that Docker and Docker Compose are installed on your host machine:

    ```bash
    docker --version
    docker-compose --version
    ```

2. Check that you have an NVIDIA driver installed on your host machine:

    ```bash
    nvidia-smi
    ```

3. Setup the NVIDIA Container Toolkit on your host machine:
   Follow the instructions bellow or from [NVIDIA's official documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

   Configure Production Repository:
   ```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```

   Optionally, configure the repository to use experimental packages:
    ```bash
    sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```

    Update the packages list from the repository:
    ```bash
    sudo apt-get update
    ```

    Install the NVIDIA Container Toolkit packages:
    ```bash
    sudo apt-get install -y nvidia-container-toolkit
    ```

    Configure the container runtime by using the nvidia-ctk command:
    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    ```

    Restart the Docker daemon:
    ```bash
    sudo systemctl restart docker
    ```

4. Check that you have CUDA installed on your host machine:
    ```bash
    nvcc --version
    ```
    If CUDA is not installed on your host machine, Install CUDA by executing the following command:
    ```bash
    sudo apt install nvidia-cuda-toolkit
    ```

### Building and Running

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/mahtaabdn/Gaussian-Heritage.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Gaussian-Heritage
    ```

3. Build and start the Docker containers:

    ```bash
    docker-compose up
    ```

### Accessing the Application

If everything works fine, you should be able to open a browser and connect to [http://127.0.0.1:5000](http://127.0.0.1:5000).


## Usage

1. Collect a set of multi-view RGB images of the scene you want to digitize
2. Use our web interface to upload images to the local server
3. The system will process the images to generate:
   - 2D instance segmentation masks
   - A sparse 3D model
   - A 3D model capturing appearances and 3D segmentation of the scene
4. Extract individual 3D models for objects of interest

A step-by-step guide with examples will be added soon.


## Citation

If you use this work in your research, please cite our paper:
