# Gaussian Heritage: 3D Digitization of Cultural Heritage with Integrated Object Segmentation

## Overview

This repository contains the implementation of "Gaussian Heritage", a pipeline for 3D digitization of cultural heritage objects using only RGB images. Our method leverages advancements in novel view synthesis and Gaussian Splatting to create 3D replicas of scenes and extract models for individual items of interest.

Accepted to the ECCV 2024 VisArt Workshop.

## Key Features

- Generate 3D replicas of scenes using only multi-view RGB images (e.g., photos from a museum)
- Extract individual 3D models for items of interest within the scene
- Deployable as a Docker container for easy setup and use
- Input can be collected using a standard smartphone


## Installation

Our pipeline is packaged as a Docker container for easy deployment. To use it:

### Prerequisites

- **Docker:** Make sure Docker is installed on your system. You can download it from [Docker’s official website](https://www.docker.com/get-started).

### Clone the Repository

Detailed installation instructions will be provided soon.


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
