# Exploring Unsupervised Learning and Dimensional Reduction
I explored unsupervised learning algorithms as well as dimensional reduction algorithms. Unsupervised learning is most often used for clustering unlabeled data to attempt to group data that have similar features and qualities together. Clustering can give one some general insight on the distribution of their data without labeling. The two unsupervised learning algorithms that will be explored are k-means clustering and expectation maximization (EM). Dimensional reduction is used to transform data into a lower dimensional space, keeping as much information as possible, to reduce computational complexity for highly computational tasks like neural network classification. The dimensional reduction algorithms used are principal component analysis (PCA), independent component analysis (ICA), randomized projections (RP), and information gain (IG). I also explore training neural networks with the reduced data, as well as using clusters as an input feature for neural network classification. All of the analysis was performed with sci-kit learn, and plots were produced with matplotlib.

## File/Code Structure
There are four jupyter notebooks that contain the code for this project.

### spam_clustering.ipynb
This notebook contains the code for the initial clustering of the original spam data.

### spam_dr.ipynb
This notebook contains basically all other code relating to the spam database. This includes the dimensional reduction, clustering the dimensionally reduced data, and running all the neural networks. All the related plotting is done on this notebook too. 

### letter_clustering.ipynb
This notebook contains the code for the initial clustering of the original letter recognition data.

### letter_dr.ipynb
This notebook contains basically all other code relating to the letter recognition database. This includes the dimensional reduction, clustering the dimensionally reduced data, and running all the neural networks. All the related plotting is done on this notebook too. 

The other python files contain helper methods for these jupyter notebooks. 

All of the plots will be outputted as .png files in this directory when you run the jupyter notebooks.
The neural network code sections save the accuracy and other metrics in text files, so the plotting can read from these text files without having to rerun the neural network training code.

Requirements:

sk-learn 0.20.1

numpy 1.15.0

matplotlib 2.2.2


Note: My analysis report is not available to prevent plagiarism. This project was done for a machine learning course at Georgia Tech.

