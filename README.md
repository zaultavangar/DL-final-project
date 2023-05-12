How to run our project:

First, to preprocess the headline dataset run preprocess.py. This will store the preprocessed headline data in the /data_preprocessed/ directory.

Next, to run the embedding model for the entire dataset, run embedding_model_full.py.
For the data split into years, run embedding_model_year_wise.py.
This will store the embeddings in the /vectors/ directory.

Then run the procrustes.py script to apply the orthogonal Procrustes method on all the embedding matrices in order to align them across years.

Use the word_analysis.py script to actually play around with the aligned embedding matrices. This file contains a method for finding nearest neighbors to a word within a given year and plotting them using the t-SNE nonlinear dimensionality reduction algorithm (plot_nearest_neighbors). It also contains a method plot_cos_similarities which plots the cosine similarity of two words of interest over all the years in which they both appear with high frequency.

Finally, to run our language models use the DL_Final_Project_Language_Model notebook included in the repository. We ran the notebook in Google Drive so that we could use GPU, and to do so you need to upload the data directories as well so the notebook can access them in training the models. Note that in the final cell of the notebook you can change the start prompt fed to the models, as well as the final index of the output of the calls to np.argsort() in order to  control the number of tokens generated (there are comments marking where to do this).

We hope you enjoy playing around with our models!