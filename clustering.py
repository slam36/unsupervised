import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import *
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import *
import sys
import warnings
warnings.filterwarnings('ignore')
RANDOM_STATE = 5000

def clustering(X, y, cluster_range, problem_name):
	km_SSE = []
	km_homogeneity = []
	km_completeness = []

	em_ll = []
	em_homogeneity = []
	em_completeness = []

	km = kmeans(random_state=RANDOM_STATE)
	gmm = GMM(random_state=RANDOM_STATE)

	for k in cluster_range:
	    km.set_params(n_clusters=k)
	    gmm.set_params(n_components=k)
	    km.fit(X)
	    gmm.fit(X)
	    y_kmeans = km.predict(X)
	    y_em = gmm.predict(X)
	    km_SSE.append(-1 * km.score(X)) #sum of squared errors
	    km_homogeneity.append(homogeneity_score(y, y_kmeans))
	    km_completeness.append(completeness_score(y, y_kmeans))
	    em_ll.append(gmm.score(X)) #log likelihood
	    em_homogeneity.append(homogeneity_score(y, y_em))
	    em_completeness.append(completeness_score(y, y_em))

	plt.figure(1)
	plt.plot(cluster_range, km_SSE)
	plt.xlabel('number of clusters')
	plt.ylabel('SSE')
	plt.title('k-means: SSE vs k')
	plt.savefig(problem_name + '_km_sse.png')

	plt.figure(2)
	plt.plot(cluster_range, km_homogeneity)
	plt.xlabel('number of clusters')
	plt.ylabel('Homogeneity')
	plt.title('k-means: Homogeneity vs k')
	plt.savefig(problem_name + '_km_homo.png')

	plt.figure(3)
	plt.plot(cluster_range, km_completeness)
	plt.xlabel('number of clusters')
	plt.ylabel('Completeness')
	plt.title('k-means: Completeness vs k')
	plt.savefig(problem_name + '_km_complete.png')

	plt.figure(4)
	plt.plot(cluster_range, em_ll)
	plt.xlabel('number of clusters')
	plt.ylabel('Log Likelihood')
	plt.title('Expectation Maximization: Log Likelihood vs k')
	plt.savefig(problem_name + '_em_ll.png')

	plt.figure(5)
	plt.plot(cluster_range, em_homogeneity)
	plt.xlabel('number of clusters')
	plt.ylabel('Homogeneity')
	plt.title('Expectation Maximization: Homogeneity vs k')
	plt.savefig(problem_name + '_em_homo.png')

	plt.figure(6)
	plt.plot(cluster_range, em_completeness)
	plt.xlabel('number of clusters')
	plt.ylabel('Completeness')
	plt.title('Expectation Maximization: Completeness vs k')
	plt.savefig(problem_name + '_em_complete.png')


