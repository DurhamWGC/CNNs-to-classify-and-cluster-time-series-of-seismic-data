# CNNs-to-classify-and-cluster-time-series-of-seismic-data
using Convolutional Neural Networks (CNNs) to classify and cluster time series of seismic data

This project introduces a comprehensive approach using a CNN-ResNet autoencoder for cluster analysis and relabeling a dataset based on the clustering labels. A CNN-ResNet classification model is then built and trained using the newly labeled training set. 
Due to computational performance limitations, this project only employed a simple method to calculate the epicentral distance. Others can refer to methods combining energy conversion calculations to better determine the P-wave arrival time. Additionally, more complex and refined autoencoder and classification models can be developed. 
The dataset used in this project consists of three-channel seismic ratio data recorded by various seismic stations during several earthquake events in Japan with magnitudes above 6.0.
