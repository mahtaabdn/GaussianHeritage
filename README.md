# Gaussian Heritage: 3D Digitization of Cultural Heritage with Integrated Object Segmentation

## Overview

This repository contains the implementation of "Gaussian Heritage", a pipeline for 3D digitization of cultural heritage objects using only RGB images. Our method leverages advancements in novel view synthesis and Gaussian Splatting to create 3D replicas of scenes and extract models for individual items of interest.

Accepted to the ECCV 2024 VisArt Workshop.

## Key Features

- Generate 3D replicas of scenes using only RGB images (e.g., photos from a museum)
- Extract individual 3D models for items of interest within the scene
- Utilizes modified Gaussian Splatting for efficient 3D segmentation
- No manual annotation required
- Input can be collected using a standard smartphone

## Installation

Our pipeline is packaged as a Docker container for easy deployment. To use it:

1. Install Docker on your system
2. Pull the Docker image (link to be provided)
3. Run the container with appropriate volume mounts for input/output

Detailed installation instructions will be provided soon.

## Usage

1. Collect a set of RGB images of the scene you want to digitize
2. Use our web interface to upload images to the local server
3. The system will process the images to generate:
   - 2D instance segmentation masks
   - A sparse 3D model
   - A 3D model capturing appearances and 3D segmentation of the scene
4. Extract individual 3D models for objects of interest

A step-by-step guide with examples will be added soon.

## Method Overview

Our pipeline consists of three key stages:
1. Model Training and Optimization
2. 2D Mask Rendering
3. 3D Extraction

For detailed information about each stage, please refer to our paper.

## Citation

If you use this work in your research, please cite our paper:
