LanTiSEAA
=========

In our paper **"Enriching Feature Engineering for Text Data by Language Time Series Analysis"**, we are extending the feature engineering approaches for text data by techniques, which have been introduced in the context of time series classification and functional data analysis. 
The general idea of the presented feature engineering approach is to tokenise the text samples under consideration and map each token to a number, which measures a specific property of the token.
Consequently, each text sample becomes a language time series, which is generated from consecutively emitted tokens and time being represented by the position of the respective token within the text sample.
The resulting language time series can be characterised by collections of established time series feature extraction algorithms from time series analysis and signal processing.
This approach maps each text sample (irrespective of its original length) to a well-defined feature space, which can be analysed with standard statistical learning methodologies.
The proposed feature engineering technique for text data is discussing for an authorship attribution problem. In the paper, we demonstrate that the extracted language time series features can be successfully combined with standard machine learning approaches for natural language processing, improves the accuracy of an authorship attribution problem, and can be used for visualizing differences and commonalities of stylometric features.

The **Language Time Series Enriched Authorship Attribution (LanTiSEAA)** is a python library implementing the methods and frameworks used in our study. It contains two main classes "LanTiSEAA" and "IterativeLanTiSEAA" along with several modules including "baseline", "buffer", "extractor" and "ts". Many of the parameters, time series mapping methods, baseline classifiers, feature extractors and so on are customizable and base classes are provided for extending our implementation by your own methods.

# Main classes and modules

## LanTiSEAA and IterativeLanTiSEAA

The "LanTiSEAA" class implemented a basic language time series enriched authorship attribution classifier. It is a classifier which can be integrated into a sklearn workflow, with the basic functionality of extracting, combining and selecting time series features from texts to enrich baseline method predictions. Sample usage of "LanTiSEAA" can be found in the jupyter notebook: ```Tutorial1_LanTiSEAA```

The "IterativeLanTiSEAA" class extends the "LanTiSEAA" and also implemented the iterative stacking framework described in the paper. The fit method of the "IterativeLanTiSEAA" captures the entire working procedure of our paper, and is capable of using a cross-validation framework to stack the time series feature groups (from different mapping methods) to the baseline method one by one, and estimate the significance of its improvement on the baseline using bayesian estimation and the Wilcoxon Signed Rank Test. In addition, IterativeLanTiSEAA will select the best combination of time series feature groups and the baseline, and fit on the entire training data set to be ready for predicting unseen data - it can also be integrated into a sklearn workflow. Sample usage of "IterativeLanTiSEAA" is described in the jupyter notebook: ```Tutorial2_IterativeLanTiSEAA```

## baseline

The "baseline" module implements five baseline Natural Language Processing based baseline methods used in our study. Customized baseline methods can either directly extend sklean's BaseEstimator and ClassifierMixin methods or the lantiseaa.baseline.BaseVectorizerClassifier class to be used in the "LanTiSEAA" and "IterativeLanTiSEAA" classes.

## buffer

The "buffer" module implements a BaseBuffer class and two concrete Buffer classes - LocalBuffer which saves/reads useful data generated during execution of "LanTiSEAA" and "IterativeLanTiSEAA" into/from hard drives on the computer; while MemoryBuffer buffers/loads data into/from the computer memory. The buffer module, especially the LocalBuffer, is specifically designed for research purpose so that all research data generated can be saved on the disk for later use. Customized buffers can extend the lantiseaa.buffer.BaseBuffer class. Some usage of LocalBuffer to read and save data from or into a local project folder can be found in the two tutorial notebooks: ```Tutorial1_LanTiSEAA``` and ```Tutorial2_IterativeLanTiSEAA```

## extractor

The "extractor" module contains the "TsfreshTSFeatureExtractor" which uses tsfresh to automatically extract statistical features from time series and select relevant features from groups of time series features. Customized feature extractors can extend the lantiseaa.extractor.BaseTSFeatureExtractor class.

## ts

The "ts" module contains the five different language time series mapper (token length frequency, token length sequence, word count vector, token frequency, token frequency rank) from two category (data set dependent or independant), that were used in our study. Customized mappers can extend either the lantiseaa.ts.BaseDatasetDependentTSTransformer or the lantiseaa.ts.BaseDatasetIndependentTSTransformer class based on the category the mapping method belongs to. ```Tutorial1_LanTiSEAA``` and ```Tutorial2_IterativeLanTiSEAA``` shows how you can choose which mapping methods to use for LanTiSEAA and IterativeLanTiSEAA.


# Installation

To use the "lantiseaa" module, cloned the repository to your local machine, and install the module using python. You can first navigate to the LanTiSEAA repository using your command line and use the following code to install the module on your local machine (or anaconda environment):

    python setup.py develop


## NLTK
Our method used some resources from nltk, please also use the NLTK Downloader to download the following resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
