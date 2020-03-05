import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


analyzer = CountVectorizer(analyzer='word').build_analyzer()
ps = PorterStemmer()
eng_stopwords = set(stopwords.words("english"))

def split_text(text):
    return [ps.stem(word) for word in analyzer(text) if word not in eng_stopwords]

class BaseVectorizerClassifier(BaseEstimator, ClassifierMixin):
    """
    Base class for NLP methods described in the paper. All these methods
    are consist of a vectorizer (for vectorizing the texts) and a classifier
    (for making predictions). 

    ...

    Attributes
    ----------
    classes_ : numpy array
        unique labels in the training data set. Retrieved from the classifier if 
        the classfier has the class_ attribute after fitting.


    Methods
    -------
    fit(self, X, y)
        Fit on the given training data set, vectorize the texts and train the classifier

    predict(self, X)
        Make predictions on the given testing data set

    predict_proba(self, X)
        Make probability predictions on the given testing data set
    
    """

    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier
    
    def fit(self, X, y):
        # train count vectorizer
        cv_train = self.vectorizer.fit_transform(X)
        # train classifier
        self.classifier.fit(cv_train, y)
        try:
            self.classes_ = self.classifier.classes_
        except AttributeError:
            self.classes_ = unique_labels(y)
        return self
    
    def predict_proba(self, X):
        # vectorization
        cv_X = self.vectorizer.transform(X)
        # prediction
        return self.classifier.predict_proba(cv_X)

    def predict(self, X):
        # vectorization
        cv_X = self.vectorizer.transform(X)
        # prediction
        return self.classifier.predict(cv_X)


######### Bag-of-words based methods #########
class BOWMNB(BaseVectorizerClassifier):

    def __init__(self, param_grid={'alpha': np.linspace(0.1, 1, 10)}, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=None), scoring='neg_log_loss', n_jobs=-1):
        super().__init__(
            vectorizer = CountVectorizer(analyzer=split_text), 
            classifier = GridSearchCV(MultinomialNB(), param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        )

class BOWSVM(BaseVectorizerClassifier):

    def __init__(self, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=None), random_state=None):
        super().__init__(
            vectorizer = TfidfVectorizer(analyzer=split_text),
            classifier = CalibratedClassifierCV(SVC(random_state=random_state), cv=cv)
        )


######### N-grams based methods #########
class CharNGrams(BaseVectorizerClassifier):

    def __init__(self, ngram_range=(1, 5), min_df=0.05, norm='l2', use_idf=True, sublinear_tf=True, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=None), random_state=None):
        super().__init__(
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range, min_df=min_df, norm=norm, use_idf=use_idf, sublinear_tf=sublinear_tf),
            classifier = CalibratedClassifierCV(SVC(random_state=random_state), cv=cv)
        )

class WordNGrams(BaseVectorizerClassifier):

    def __init__(self, ngram_range=(1, 3), norm='l2', use_idf=True, sublinear_tf=False, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=None), random_state=None):
        super().__init__(
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, norm=norm, use_idf=use_idf, sublinear_tf=sublinear_tf),
            classifier = CalibratedClassifierCV(SVC(random_state=random_state), cv=cv)
        )


######### Character Sequence Kernel with SVM #########
def kernel_object(X, Y, n_grams=[1, 2, 3]):
    weight = 0
    
    X_split = {}
    Y_split = {}
    
    Q_star = set()
    
    for i in n_grams:
        for x in ngrams(list(X), i):
            if x not in X_split:
                X_split[x] = 1
            else:
                X_split[x] += 1
            Q_star.add(x)
            
        for y in ngrams(list(Y), i):
            if y not in Y_split:
                Y_split[y] = 1
            else:
                Y_split[y] += 1
            Q_star.add(y)
    
    for q in Q_star:
        try:
            weight += X_split[q] * Y_split[q]
        except KeyError:
            pass
    
    return weight

def kernel_normalized(X, Y, n_grams=[1, 2, 3]):
    return kernel_object(X, Y, n_grams) / np.sqrt(kernel_object(X, X, n_grams) * kernel_object(Y, Y, n_grams))

def kernel_matrix(X, Y, n_grams=[1, 2, 3]):
    kernels = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            kernels[i][j] = kernel_normalized(X[i], Y[j], n_grams)
    return kernels

class CSKSVM():
    ''' The implementation of Conrad Sanderson and Simon Guenter's Sequence Kernel method

    This class implements the character sequence kernel method used by Conrad Sanderson and 
    Simon Guenter in their paper "Short Text Authorship Attribution via Sequence Kernels, 
    Markov Chains and Author Unmasking: An Investigation" [1] to predict authorships of 
    short texts.

    [1] Sanderson, C., Guenter, S.: Short text authorship attribution via sequence kernels, 
        markov chains and authorunmasking: An investigation. In: Proceedings of the 2006 
        Conference on Empirical Methods in NaturalLanguage Processing, pp. 482–491 (2006). 
        Association for Computational Linguistics

    '''

    def __init__(self, n_grams=[1,2,3]):
        self.n_grams = n_grams

    def fit(self, X, y, sample_splits=100, random_state=42):
        # take sample train and validate sets
        if sample_splits < 2:
            raise Exception("sample_splits cannot be smaller than 2")
        self.validation_samples = None
        self.y_validate = None
        self.train_samples = None
        self.y_train_sample = None
        skf = StratifiedKFold(n_splits=sample_splits, shuffle=True, random_state=random_state)
        for _, index in skf.split(X, y):
            if self.validation_samples is None: # give the first fold test to validate data set
                self.validation_indices = index
                self.validation_samples = X[index]
                self.y_validate = y[index]
            # give the last fold test to train data set
            self.train_indices = index
            self.train_samples = X[index]
            self.y_train_sample = y[index]
        
        # calculate train and validate kernels
        self.kernel_train = kernel_matrix(self.train_samples, self.train_samples)
        self.kernel_validate = kernel_matrix(self.validation_samples, self.train_samples)

        # train SVC and CalibratedClassifierCV
        self.svc = SVC(kernel='precomputed', random_state=random_state)
        self.svc.fit(self.kernel_train, self.y_train_sample)
        self.classifier = CalibratedClassifierCV(self.svc, method="sigmoid", cv="prefit")
        self.classifier.fit(self.kernel_validate, self.y_validate)
        
        return self

    def predict_proba(self, X, return_kernels=False):
        # calculate kernels
        kernel_test = kernel_matrix(X, self.train_samples)
        self.kernel_test = kernel_test

        # prediction
        pred = self.classifier.predict_proba(kernel_test)
        if return_kernels:
            return pred, kernel_test
        else:
            return pred

    def predict(self, X, return_kernels=False):
        # calculate kernels
        kernel_test = kernel_matrix(X, self.train_samples)
        self.kernel_test = kernel_test

        # prediction
        pred = self.classifier.predict(kernel_test)
        if return_kernels:
            return pred, kernel_test
        else:
            return pred