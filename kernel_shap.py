import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import itertools
import os
import re
import scipy as sp
from functools import partial
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge
from scipy.special import softmax

class LimeBase:
    def __init__(self, kernel_fn, random_state=None):
        self.kernel_fn = kernel_fn
        self.random_state = check_random_state(random_state)
    
    def explain_instance_with_data(self, neighborhood_data, neighborhood_labels, distances, label, num_features, model_regressor=None): 
        """
        The function generates explanation based on the perturbed data.
        """
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]

        # Select the set of features that is most likely to be the important one.
        # used_features = self.feature_selection(neighborhood_data, labels_column, weights, num_features)

        # The definition of a simple and interpretable model.
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
        
        easy_model = model_regressor
        easy_model.fit(neighborhood_data, labels_column, sample_weight=weights)

        #prediction_score = easy_model.score(neighborhood_data, labels_column, sample_weight=weights)
        featurs2importance = list(zip(list(range(neighborhood_data.shape[-1])), easy_model.coef_))
        featurs2importance = sorted(featurs2importance, key=lambda x: np.abs(x[1]), reverse=True)
        return featurs2importance


    
    def feature_selection(self, data, labels, weights, num_features):
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        
        for _ in range(min(num_features, data.shape[1])):
            max_ = -1 * np.inf
            best = 0
            for feature in range(data.shape[-1]):
                if feature in used_features:
                    continue
                
                clf.fit(data[:, used_features + [feature]], labels, sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]], labels, sample_weight=weights)

                if score > max_:
                    best = feature
                    max_ = score 
            used_features.append(best)
        return np.array(used_features)


class IndexedString:
    """
    The class is used for converting between an interpretable representation and the original text.
    """
    def __init__(self, test_instance, split_expression=r'\W+', bow=True):
        self.raw_string = test_instance
        splitter = re.compile(r'(%s)|$' % split_expression)
        self.as_list = [s for s in splitter.split(test_instance) if s]
        self.as_np = np.array(self.as_list)
        
        string_start = np.hstack(([0], np.cumsum([len(x) for x in self.as_np[:-1]])))
        vocab = {} # word to id
        self.inverse_vocab = [] # id to word
        self.positions = [] # id to word position list
        self.bow = bow

        for i, word in enumerate(self.as_np):
            if splitter.match(word):
                continue
            if bow:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = vocab[word]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(positions)
        
        self.num_words = len(self.inverse_vocab)
    
    def inverse_removing(self, words_to_remove):
        """
        According to 'words_to_remove' remove all the elements.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            pass
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])
    
    def __get_idxs(self, words):
        if self.bow:
            return list(itertools.chain.from_iterable([self.positions[z] for z in words]))
        else:
            return self.positions[words]
class LimeTextExplainer:
    """
    The class generates explanation for text.
    """
    def __init__(self, kernel_width=25, class_names=None, random_state=1):
        self.class_names = class_names
        self.random_state = check_random_state(random_state)

        def kernel(d, kernel_width=kernel_width):
            return np.sqrt(np.exp(-(d**2) / kernel_width ** 2))
        
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.inf_weight = 1000000.0
        self.base = LimeBase(kernel_fn, random_state=self.random_state)

    def explain_instance(self, test_instance, classifier_fn, labels=(1,), num_features=10, num_samples=5000, distance_metric='cosine', model_regressor=None):

        indexed_string = IndexedString(test_instance)

        z_data, z_label, distances = self.shap_data_labels_distances(indexed_string, classifier_fn, num_samples, distance_metric=distance_metric)

        for label in labels:
            explanation = self.base.explain_instance_with_data(z_data, z_label, distances, label, num_features)
        self.print_explanation(explanation, indexed_string)
    
    def print_explanation(self, explanation, indexed_string):
        for idx, score in explanation:
            print(indexed_string.inverse_vocab[idx], '\t', score)

    def shap_data_labels_distances(self, indexed_string, classifier_fn, n_samples, **kwargs):
        """
        It is the function to generate perturbed data, where the classifier_fn is the black-box model to be explained.
        """
        n_features = indexed_string.num_words
        inf_weight = 1000000.0
        # distance_fn is the function to define the distance between the perturbed samples and the original sample.
        def distance_fn(n_active_feature_list):
            if n_features == 1: # only one features to be explained.
                dis_list = np.ones(len(n_active_feature_list))
                return dis_list
            
            dis_list = list()
            for n_active_features in n_active_feature_list:
                if n_active_features == 0 or n_active_features == n_features:
                    dis = self.inf_weight
                else:
                    dis = 1.0
                dis_list.append(dis)
            dis_list = np.array(dis_list)
            return dis_list

        n_interp_features = n_features
        # the number of remained words for each examples.
        n_features_list = np.arange(n_features, dtype=np.float32)
        denom = n_features_list * (n_interp_features - n_features_list)
        probs_nonzeros = (n_interp_features - 1) / denom

        probs_nonzeros[0] = 0.0
        probs_nonzeros = softmax(probs_nonzeros)

        n_remained_features_list = list()
        
        #sample = self.random_state.randint(1, doc_size+1, num_samples - 1)
        data = np.ones((n_samples, n_features))
        data[0] = np.ones(n_features)
        data[1] = np.zeros(n_features)
        features_range = range(n_features)
        inverse_data = ['', indexed_string.raw_string]
        n_remained_features_list.append(0)
        n_remained_features_list.append(n_features)
        
        for i in range(2, n_samples, 1):
            n_active = self.random_state.choice(features_range, 1, p=probs_nonzeros)
            probs = np.random.randn(len(probs_nonzeros))
            # obtain k-th smallest value 
            kthvalue = np.partition(probs, n_active)[n_active]

            n_remained_features_list.append(n_active)
            remove_index = np.argwhere(probs >= kthvalue).astype(int).flatten()
            # obtain z^{'}
            data[i, remove_index] = 0
            # z^{'} to z
            inverse_data.append(indexed_string.inverse_removing(remove_index))
            
        labels = classifier_fn(inverse_data)
        distances = distance_fn(n_remained_features_list)
        return data, labels, distances        

if __name__ == "__main__":
    np.random.seed(0)
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    class_names = ['atheism', 'christian']

    # transform raw data to vectors(TF-IDF)
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(newsgroups_train.data)
    test_vectors = vectorizer.transform(newsgroups_test.data)
    
    # train model to predict.
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    rf.fit(train_vectors, newsgroups_train.target)

    pred = rf.predict(test_vectors)


    f1 = sklearn.metrics.f1_score(newsgroups_test.target, pred)
    acc = sklearn.metrics.accuracy_score(newsgroups_test.target, pred)
    print("f1_score", f1)
    print("acc", acc)

    c = make_pipeline(vectorizer, rf)
    print(c.predict_proba([newsgroups_test.data[0]]))

    # The generation of explanation.
    explainer = LimeTextExplainer(class_names=class_names)
    idx = 83
    exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)

