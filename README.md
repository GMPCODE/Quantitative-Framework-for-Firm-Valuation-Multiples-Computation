# Portfolio Optimization Techniques through Deep Reinforcement Learning Models

## Abstract

I develop a portfolio optimization model that periodically adjusts the weights over a period of time. I implement different machine learning techniques to compare the performances over different metrics. The models have been trained over two different edited versions of the 12 industries portfolio dataset of daily returns by Eugene Fama and Kenneth R. French. These two versions of the datasets contain statistical information computed over the original dataset, such as the standard deviation and the momentum. One of these two versions includes information relative to the sentiment analysis performed all over the 12 industries using a large language model. The models are designed in different versions, implementing either convolutional neural networks or recurrent neural networks. The convolutional neural network shows positive results when compared to a portfolio through a mean-variance optimization approach. In some cases, models implementing convolutional layers outperform those built on recurrent layers. Some of the findings show better performances with models trained through the dataset containing the sentiment analysis for the industries. My findings suggest that the models based on the convolutional layers perform effectively when compared to classical portfolio optimization approaches.

## Thesis

[PDF Available]

## Development

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)


This __repository__ hosts the code related to my __Bachelor Thesis__, which aims at offers new approaches to __*portfolio optimization*__.<br>
This __project__ explores the intersection of __*Deep Learning*__ and __*Reinforcement Learning*__ __techniques__ to elaborate a novel way of optimizing portfolios.<br>
