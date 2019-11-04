# Unsupervised Learning and Dimensional Reduction

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

