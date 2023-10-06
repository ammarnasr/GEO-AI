# Cropland Extent Mapping Challenge

This repository contains my submission for the Cropland Extent Mapping Challenge, where I have developed a cropland extent mapping solution using machine learning techniques. This README.md file provides an overview of my project and how to use the code.

## Project Overview

In this challenge, I aim to create accurate and cost-effective classification models for cropland extent mapping with a spatial resolution of 10 meters. The challenge also involves testing the temporal extendibility of the proposed method at a local scale. The evaluation will be based on accuracy and will consider various aspects, including comparison with existing cropland maps and the novelty and practicality of the procedure.

## Code Structure

My code in this repository performs the following tasks:

1. **Data Retrieval**: I fetch Sentinel-2 LA satellite data for the required regions.
2. **Region Masking**: I apply region-specific masks to focus on Sudan, Iran, and Afghanistan.
3. **Training**: I train a stacked classifier for cropland extent mapping.
4. **Inference**: I generate cropland extent maps for the target regions using the trained model.

## Getting Started

To use my code and replicate the results, follow these steps:

1. **Install Dependencies**: Make sure you have all the necessary dependencies installed. You can find the list of dependencies in the `requirements.txt` file.

2. **Data Preparation**: Prepare the required data, including satellite imagery and ground truth data. Or you can use the provided pre-processed data geodataframes saved in the `gdfs/` directory as `{country}_{regoin}_processed_gdf.joblib`.

3. **Training**: Train the machine learning model using the provided training dataset. Or you can use the provided pre-trained model saved in the `models/` directory as `{region}_model.joblib`.

4. **Inference**: Generate cropland extent maps for Sudan, Iran, and Afghanistan using the trained model.

## Sample Inference Maps

You can find sample inference maps in the `samples/` directory. These maps showcase the results of my cropland extent mapping solution for Sudan, Iran, and Afghanistan.
#### Sudan: Cropland Extent Map
![Sudan Cropland Extent Map](samples/Sudan_cropland_extent_map.png)

## Cloud Coverage Analysis

I have also conducted an analysis of cloud coverage over time for the three areas. The results of this analysis are available in the `samples/` directory, where you can find plots and statistics on cloud coverage.
#### Sudan: Cloud Coverage Statistics
![Sudan Cloud Coverage](samples/Sudan_cloud_coverage_2019.png) 

## Evaluation

I am participating in this challenge to contribute to the advancement of global cropland mapping. My submission will be evaluated based on accuracy and other criteria specified by the challenge organizers.

## Contact

If you have any questions or feedback regarding my submission, please feel free to reach out to me:
- [Ammar Khairi](mailto:ammarnasraza@gmail.com)

Thank you for your interest in my project!

