import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from functools import partial
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import VotingClassifier




