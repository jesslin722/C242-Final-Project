# Leveraging CNN-LSTM Autoencoder and GNN Classifier for Predicting Potential HIV Therapeutic Molecules
Course BioE C242- Machine Learning, Statistical Models, and Optimization for Biological and Chemical Problems Final Project

U.C. Berkeley– Spring 2024

Members: 
Yu-Chieh (Jess) Lin | jesslin.722@berkeley.edu, Yi-Hong (Ian) Liu | yihong_liu@berkeley.edu

Instructor: 
Professor Teresa Head-Gordon | thg@berkeley.edu

GSI consultant: 
Yingze (Eric) Wang | ericwangyz@berkeley.edu

## Introduction 
In this project, we first used CNN-LSTM combined with the autoencoder to generate a new possible candidate molecule. Next, we trained a GNN model to help us distinguish whether this new molecule is active or inactive.

## Datasets
The Drug Therapeutics Program (DTP) AIDS Antiviral Screen developed the HIV dataset, assessing over 40,000 compounds for their efficacy in impeding HIV replication. The outcomes of this screening were classified into three distinct groups: compounds confirmed as inactive (CI), those confirmed as active (CA), and those identified as moderately active (CM). Furthermore, HIV inactive, including only the inactive compounds, has 39,684 molecules. HIV active, which combines both the moderately active and active compounds, accounting for a total of 1,443 molecules.

However, a major challenge arises from this dataset- Imbalanced Class Distribution. The distribution of classes is highly skewed towards inactive compounds. This imbalance can lead to a model bias towards predicting compounds as inactive. So we have to balance our dataset before training through appropriate sampling, this will help our model learn to recognize subtle differences between active and inactive compounds, rather than overwhelmingly learning from the majority class.

## Methods
1. CNN-LSTM Autoencoder
- Data preprocessing and sampling:
  - Our dataset originally represents the molecules via SMILES strings. We changed it into a SELFIES string. SELFIES provided a 100% robust molecular string representation. Ｗe changed them all into SELFIES and made the analysis. Next, we sampled 400 molecules for each category and tokenized the string with the index value of the vocabularies of the sampled dataset. But sequence length varied string by string, so we padded each data to ensure a consistent length before we sent the data into our model.
- Model:
  - Our model cited from the paper, Convolutional, Long Short-Term Memory, fully connected Deep Neural Networks, which was published by Google in 2015. In this paper, they set up convolutional layers before the LSTM layers. Convolutional layers helped deal with the frequency of the elements in string and LSTM layers were good at temporal series feature extraction. Google used this model to test the performance of speech recognition and machine translation systems.
  - We used nn.Linear to increase the dimensionality of feature channels. Then, the output went through 1 dimension convolutional layer. Last, we used LSTM to generate the hidden space and use nn.Linear to transform it to the latent space of the autoencoder.
  - The decoder had the same structure but the data flowed backward.
- Result:
  - When the training task was done, we obtained an autoencoder model that could extract the features from strings and reconstruct the SELFIES string with the feature vectors. We then worked on the generation part.
  - We sampled 150 active molecules from the dataset and sent them into the encoder. From the encoder, we can acquire the feature vectors of the SELFIES string of these 150 active molecules. Each active molecule had feature vectors whose dimensions were 64. All feature vectors are composed of a distribution in each dimension. We picked up values from each distribution and made them a new feature vector.
  - Last, we sent these new vectors into a decoder to transform them into SELFIES strings and detokenized SELFIES strings. Our model can reconstruct the feature vectors that are unseen in the original dataset and they are valid molecules.

2. GNN Classifier
- Data preprocessing and sampling:
  - For validation, we extracted 50 samples from each activity class. The remaining molecules are used as training data. To address the imbalance between inactive and active classes, we employed a custom data loader during training that dynamically selects 1,500 inactive and 1,000 active samples for each batch. Prevent our model from becoming biased toward the majority class.
- Conversion of SMILES to graph:
  - The second step was converting SMILES strings into graph-based structures. Each atom in the SMILES string became a node in the graph, and each bond translated into an edge. We extracted features such as atomic number, charge, and bond type, essential for understanding the molecular structure.
  - Using the Weisfeiler-Lehman algorithm, we iteratively updated atom features by aggregating information from neighboring atoms. Enrich our node features by incorporating local chemical environment data.
  - Before feeding data into our GNN, we normalized the feature vectors through the padding, adding zeros to the shorter feature vectors to ensure all nodes have the same number of features and fixed-size input vectors.
- Model:
  - Input node feature, edge index, and feature vector into the GNN model. Transform initial node and edge features into higher-dimensional space. The GNN employed a core message-passing mechanism, where information from neighboring nodes was aggregated through edges, enabling each node to update its features based on both its properties and its interactions with adjacent nodes also for the edge.
  - Finally, the GNN aggregated all node features to form a single graph-level representation, used in the readout layer to classify the entire molecule as either HIV active or inactive. 
