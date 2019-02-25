import os
import itertools
import numpy as np
import pandas as pd
import pickle
from tabulate import tabulate
import matplotlib.pyplot as plt
from numpy.random import seed #to set random seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier

class FraudClassifier():
    '''
    Class for handling model building and new data classification
    '''

    def __init__(self, df_path):
        self.df_path = df_path # where input data lives
        self.random_seed = 1
        self.results_df = pd.DataFrame() # df for storing model results (empty for now)
        # self.models = [LogisticRegression, MultinomialNB, RandomForestClassifier] # non CNN models to fit
        # self.model_names = ['LogReg', 'MNB', 'Random_forest'] # Names of models
        self.models = [LogisticRegression, RandomForestClassifier, GradientBoostingClassifier] # non CNN models to fit
        self.model_names = ['LogReg', 'Random_forest', 'GBR'] # Names of models
        self.metrics = [accuracy_score, precision_score, recall_score]
        self.metric_names = ['accuracy', 'precision', 'recall']
        self.n_trees = 100
        self.img_dir = '../images'

    def run_analysis(self):
        print('Loading data')
        self.read_data()
        print('Spitting data')
        self.split_train_test()
        print('Creating image directory')
        self.make_dir(self.img_dir)
        print('Running Models')
        self.run_models()
        # self.plot_roc_curves(self.models, self.X_test, self.y_test)

    def read_data(self):
        self.df = pd.read_csv(self.df_path)
        keep_cols = [x for x in self.df.columns if not x.startswith('Unnamed:')]
        self.df = self.df[keep_cols]

    def split_train_test(self):
        self.y = self.df.pop('fraud').values
        self.X = self.df.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                        random_state=self.random_seed, stratify=self.y)

    @staticmethod
    def make_dir(directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    def run_models(self):
        for model, m_name in zip(self.models, self.model_names):
            setattr(self, m_name, model)
            results = pd.Series()
            results['Model'] = m_name
            self.model = getattr(self, m_name)
            self.model = self.make_model(self.model)
            for score, s_name in zip(self.metrics, self.metric_names):
                self.threshold = self.threshold_selector(self.model,
                                                            m_name,
                                                            accuracy_score, #ALL MODELS OPTIMIZING FOR ACC
                                                            s_name)
                y_pred_best = (self.model.predict_proba(self.X_test)[:, 1] >= self.threshold) # setting the threshold
                results[s_name] = score(self.y_test, y_pred_best)
                results['log_loss'] = log_loss(self.y_test, y_pred_best)
                self.save_cm(y_pred_best, m_name + '_' + s_name)
                results[s_name+'_threshold'] =self.threshold
            setattr(self, m_name, self.model)
            self.results_df = self.results_df.append(results, ignore_index=True)
        self.plot_roc_curves()

    def make_model(self, classifier, **kwargs):
        '''
        Make specified sklearn model
        args:
        classifier (object): sklearn model object
        X_train (2d numpy array): X_train matrix from train test split
        y_train (1d numpy array): y_train matrix from train test split
        **kwargs (keyword arguments): key word arguments for specific sklearn model
        '''
        model = classifier(**kwargs)
        model.fit(self.X_train, self.y_train)
        return model

    def threshold_selector(self, classifier, classifier_name, score_type, score_name):

        '''
        Find optimal threshold for given score type
        args:
        classifier (object): model
        classifier_name (string): string name of classifier
        score_type (object): sklearn metric object
        score_name (string): string name of score_type
        X_test (2d numpy array): 2d array of X values from train test split
        y_test (1d numpy array): 1d array of targets from train test split
        '''
        score = []
        thresholds = list(np.arange(0, 1.01, 0.01))
        for threshold in thresholds:
            y_pred = (classifier.predict_proba(self.X_test)[:, 1] >= threshold) # setting the threshold
            score.append(score_type(self.y_test, y_pred))
        plt.plot(thresholds, score)
        plt.xlabel('Threshold')
        plt.ylabel(score_name)
        plt.title('{}: {} vs. threshold'.format(classifier_name, score_name))
        plt.savefig(self.img_dir+'/'+classifier_name+ '_'+ score_name +'_threshold.png')
        plt.close()
        return thresholds[np.argmax(score)]

    def plot_roc(self, fitted_model, ax):
        probs = fitted_model.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, probs[:,1])
        auc_score = round(roc_auc_score(self.y_test,probs[:,1]), 4)
        ax.plot(fpr, tpr, label= f'{fitted_model.__class__.__name__} = {auc_score} AUC')

    def plot_roc_curves(self):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        for m_name in self.model_names:
            self.plot_roc(getattr(self, m_name), ax)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Luck')
        ax.set_xlabel("False Positive Rate (1-Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity, Recall)")
        ax.set_title("ROC plot of 'Fraud, Not Fraud'")
        ax.legend()
        plt.savefig(self.img_dir+'/ROC.png')
        # plt.show()
        plt.close()

    def save_cm(self, y_pred, output):
        cnf_matrix = confusion_matrix(self.y_test, y_pred.T)
        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, ['Not Fraud', 'Fraud'], normalize=True,
                              title=output)
        outfile = '../images/'+output+'.png'
        plt.savefig(outfile)
        # plt.show()
        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    @staticmethod
    def write_model(model_name, file_name='model.pkl'):
        with open(file_name, 'wb') as f:
            # Write the model to a file.
            pickle.dump(model_name, f)

    @staticmethod
    def to_markdown(df, round_places=3):
        """Returns a markdown, rounded representation of a dataframe"""
        print(tabulate(df.round(round_places), headers='keys', tablefmt='pipe', showindex=False))

if __name__=='__main__':
    # home = '/Users/paulsandoval/Documents/galvanize-dsi/m3_cloud_computing/dsi-fraud-detection-case-study/'
    df_path = '../data/train_df.csv'
    fc = FraudClassifier(df_path)
    fc.run_analysis()
    fc.to_markdown(fc.results_df)
    fc.write_model(fc.GBR) #NOTE: Model can be change to find best model
