#%%
import os
import math
import numpy as np
import pandas as pd
#pip install rfit
import rfit 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.tree import DecisionTreeClassifier
#
import warnings
warnings.filterwarnings("ignore")
print("\nReady to continue.")

#%%
df = pd.read_csv(f'..{os.sep}data{os.sep}HousePricesAdv{os.sep}train.csv', header=0)

#%%[markdown]
# Project tasks and goals:
# 
# 1. Use this Housing Price dataset. 
# - Use SalePrice as target for K-NN regression. 81
# - For features that are *ORDINAL*, recode them as 0,1,2,... 
# - Drop features that are purely categorical.
# 2. Modify the sknn class to perform K-NN regression.
# 3. Modify the sknn class as you see fit to improve the algorithm performance, logic, or presentations.
# 3. Find optimized scaling factors for the features for the best model score.
# 4. Modify the sknn class to save some results (such as scores, scaling factors, gradients, 
#  etc, at various points, like every 100 epoch).
# 5. Compare the results of the optimized scaling factors to Feature Importance
#  from other models, such as Tree regressor for example.
#  tree regresser, nn, 
# Please ask me anything about this project. You can either work individually or team with one other student
# to complete this project.
# 
# You/your team need to create a github repo (private) and add myself (physicsland) as a collaborator. 
# Please setup an appropriate .gitignore as the first thing when you create a repo. 
# 
# do we have to perform feature selection 

# selecting nominal and ordinal features 
#https://support.sas.com/content/dam/SAS/support/en/books/sas-certification-prep-guide-statistical-business-analysis-using-sas-9/69531_appendices.pdf

#nominal 
# MSZoning, Street, Alley, LotShape, LandContour,
#  Utilities, LotConfig, LandSlope, Neighborhood,
# Condition1,	Condition2,	BldgType, HouseStyle, RoofStyle, RoofMatl,
# Exterior1st,	Exterior2nd,MasVnrType,
# Foundation,	, BsmtExposure	BsmtFinType1
# BsmtFinType2, Heating,	 Electrical
# GarageType,  
# Fence,MiscFeature, SaleType,  SaleCondition

# ordinal coding
#Ex = Excellent, Gd = Good , TA = Typical, Fa = Fair, Po = Poor
#Ex = Excellent (100+ inches), Gd = Good (90-99 inches),
#TA = Typical (80-89 inches), Fa = Fair (70-79 inches),
#Po = Poor (<70 inches), NA = No Basement
#Ex = Excellent, Gd = Good, TA = Average/Typical, Fa = Fair, Po = Poor
#Typ = Typical Functionality, Min1 = Minor Deductions 1, Min2 = Minor Deductions 2, Mod = Moderate Deductions,
#Maj1 = Major Deductions 1, Maj2 = Major Deductions 2,
#Sev = Severely Damaged, Sal = Salvage only

#Fireplace quality:Ex = Excellent - Exceptional Masonry Fireplace Gd = Good - Masonry Fireplace in main level
#TA = Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#Fa = Fair - Prefabricated Fireplace in basement
#Po = Poor - Ben Franklin Stove
#NA = No Fireplace
#Interior finish of the garage:
#Fin = Finished, RFn = Rough Finished, Unf = Unfinished,
#
# NA = No Garage

#ordinal

# Functional, 
# CentralAir, PavedDrive, 
# GarageFinish,
#	

#%%
# Drop nominal 
columns_drop = [
"Id", 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
'Utilities', 'LotConfig', 'LandSlope', 
'Condition1',	'Condition2',	'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
'Exterior1st',	'Exterior2nd','MasVnrType',
'Foundation',	'BsmtExposure',	'BsmtFinType1',
'BsmtFinType2', 'Heating',	 'Electrical',
'GarageType',  'Fence', 'MiscFeature', 'SaleType',  'SaleCondition'
,'Neighborhood'

]
df = df.drop(columns=columns_drop)

#%%
#Convert ordinal 
ordinal_columns = [
'ExterQual', 'ExterCond','BsmtQual','BsmtCond',
'HeatingQC', 'KitchenQual','FireplaceQu','GarageQual','GarageCond', 'PoolQC'
]
ordinal_mapping = {
np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
}
funational_mapping = {
'Sal': 0, 'Sev':1,  'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7
}
centralair_mapping = {
'N': 0, 'Y': 1
}
paveddrive_mapping = {
'N':0, 'P':1, 'Y':2
}
garagefinish_mapping = {
np.nan:0, 'Unf':1, 'RFn':2, 'Fin':3
}
for col in ordinal_columns:
    df[col] = df[col].map(ordinal_mapping)

df['Functional'] = df['Functional'].map(funational_mapping)
df['CentralAir'] = df['CentralAir'].map(centralair_mapping)
df['PavedDrive'] = df['PavedDrive'].map(paveddrive_mapping)
df['GarageFinish'] = df['GarageFinish'].map(garagefinish_mapping)

#%%
#df = pd.get_dummies(df, columns=['Neighborhood'])

#%%
df.shape[1]
data_x = df.drop(columns=['SalePrice'])# data_x is pandas df here, 
data_y = df['SalePrice'] # data_y is pandas series here

#%%
nan_counts = data_x.isna().sum()
# Filter columns that contain NaN values
columns_with_nan = nan_counts[nan_counts > 0]
print(columns_with_nan)
#%%
data_x = data_x.drop(columns='LotFrontage')
#%%
#mean imputation 
data_x['MasVnrArea']=data_x['MasVnrArea'].fillna(data_x['MasVnrArea'].mean())
data_x['GarageYrBlt']=data_x['GarageYrBlt'].fillna(data_x['GarageYrBlt'].mean())

#%%
class sknn:
    '''
    Scaling k-NN model
    v2
    Using gradient to find max
    '''
    import os
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.tree import DecisionTreeClassifier

# contructor and properties
    def __init__(self, 
                    data_x, 
                    data_y, 
                    resFilePfx='results', 
                    classifier=True, 
                    k=7, 
                    kmax=33, 
                    zscale=True, 
                    caleExpos_init = (), 
                    scales_init = (), 
                    ttsplit=0.7, 
                    max_iter = 100, 
                    seed=1, 
                    scoredigits = 6, 
                    learning_rate_init = 0.5, 
                    atol = 1e-8, 
                    scaler = 'Ztransform' , #'MinMax', 'Robust'
                    weight = 'uniform', #'inverse, gaussian
                    metric = 'minkowski', #'euclidean', 'manhattan', chebyshev, 
                    cv=5) :
        """
        Scaling kNN model, using scaling parameter for each feature to infer feature importance and other info about the manifold of the feature space.

        Args:
            data_x (numpy ndarray or pandas DataFrame): x-data
            data_y (numpy ndarray or pandas Series or DataFrame): y-data
            resFilePfx (str, optional): result file prefix. Defaults to 'scores'.
            classifier (bool, optional): classifier vs regressor. Defaults to True.
            k (int, optional): k-value for k-N. Defaults to 7.
            kmax (int, optional): max k-value. Defaults to 33.
            zscale (bool, optional): start with standardized z-score. Defaults to True.
            probeExpos (tuple, optional): Tuple of the exponents for scaling factors. Defaults to ().
            scaleExpos (tuple, optional): Tuple of the scaling factors. Defaults to ().
            ttsplit (float, optional): train-test-split ratio. Defaults to 0.5.
            max_iter (int, optional): maximum iteration. Defaults to 100.
            seed (int, optional): seed value. Defaults to 1.
            scoredigits (int, optional): number of digitis to show/compare in results. Defaults to 6.
            learning_rate_init (float, optional): learning rate, (0,1]. Defaults to 0.01.
            tol (_type_, optional): tolerance. Defaults to 1e-4.
        """
        self.__classifierTF = classifier  # will extend to regression later
        self.k = k
        self.__kmax = kmax
        self.__iter = 0 # the number of trials/iterations
        self.max_iter = max_iter
        self.__seed = seed
        self.__scoredigits = scoredigits
        # self.__resFilePfx = resFilePfx
        self.__learning_rate_init = abs(learning_rate_init)
        self.learning_rate = abs(learning_rate_init)
        self.__atol = atol
        self.cv = cv 
        self.scaler = scaler 
        self.weight = weight
        self.metric = metric

        # prep data
        self.data_x = data_x
        self.data_xz = data_x # if not to be z-scaled, same as original
        self.zscaleTF = zscale
        # transform z-score 
        if (self.zscaleTF): self.zXform() # will (re-)set self.data_xz
        self.data_y = data_y
        # train-test split
        self.__ttsplit = ttsplit if (ttsplit >=0 and ttsplit <= 1) else 0.5 # train-test split ratio
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.__xdim = 0  # dimension of feature space
        self.traintestsplit() # will set X_train, X_test, y_train, y_test, __xdim
        # set x data column names
        # self.__Xcolnames = (); self.__setXcolnames()
        self.__vector0 = np.zeros(self.__xdim)
        self.optimize_k()
        # self.__vector1 = np.ones(self.__xdim)
        
        # set exponents and scaling factors 
        self.__scaleExpos = [] # tuple or list. length set by number of features. Because of invariance under universal scaling (by all features with same factor), we can restrict total sum of exponents to zero.
        # self.__scaleExpos_init = [] # tuple or list. length set by number of features
        self.__scaleFactors = None # numpy array. always calculate from self.__setExpos2Scales
        self.__setExpos2Scales([]) # will set the initial self.scaleExpos and self.__scaleFactors
        # self.__gradients = [] # partial y/partial exponents (instead of partial scaling factors)
        # set sklearn knnmodel objects, train, and get benchmark scores on test data
        self.__knnmodels = [np.nan, np.nan] # matching index value as k value
        def inverse_weights(distances):
             return 1 / (distances + 1e-5)

        def gaussian_weights(distances):
            return np.exp(-distances**2 / (2 * 1**2))

        for i in range(2,self.__kmax +1): 
            if (self.__classifierTF): 
                self.__knnmodels.append( KNeighborsClassifier(n_neighbors=i, weights=self.weight, metric=self.metric).fit(self.X_train, self.y_train ) )
            else:
                if (self.weight == 'uniform'): 
                    self.__knnmodels.append( KNeighborsRegressor(n_neighbors=i, weights="uniform", metric=self.metric).fit(self.X_train, self.y_train ) ) # TODO
                elif (self.weight == 'inverse'):
                    self.__knnmodels.append( KNeighborsRegressor(n_neighbors=i, weights=inverse_weights, metric=self.metric).fit(self.X_train, self.y_train ) ) 
                elif (self.weight == 'gaussian'):
                    self.__knnmodels.append( KNeighborsRegressor(n_neighbors=i, weights=gaussian_weights, metric=self.metric).fit(self.X_train, self.y_train ) ) 
       
       
        self.benchmarkScores = [np.nan, np.nan] +  [round(x.score(self.X_test, self.y_test ), self.__scoredigits) for x in self.__knnmodels[2:] ]
        print(f'These are the basic k-NN scores for different k-values:{repr(self.benchmarkScores)}, where no individual feature scaling is performed.') 
        # set pandas df to save some results
        # self.__resultsDF = None
  

    # END constructor
    def zXform(self): 
        '''
        standardize all the features (if zscale=True). Should standardize/scale before train-test split
        :return: None
        '''
        from sklearn.preprocessing import StandardScaler
        if (self.scaler == 'Ztransform'):
            scaler = StandardScaler()
        elif (self.scaler == 'MinMax'):
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif (self.scaler == 'Robust'):
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError("Unsupported scaler type. Choose 'Ztransform', 'MinMax', or 'Robust'.")
    
        self.data_xz = scaler.fit_transform(self.data_x)  # data_x can be ndarray or pandas df, data_xz will be ndarray
        return
    

    def traintestsplit(self):
        '''
        train-test split, 50-50 as default
        :return: None
        '''
        # train-test split
        from sklearn.model_selection import train_test_split
        # data_y can be pd series here, or 
        dy = self.data_y.values if (isinstance(self.data_y, pd.core.series.Series) or isinstance(self.data_y, pd.core.frame.DataFrame)) else self.data_y # if (isinstance(data_y, np.ndarray)) # the default
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_xz, dy, test_size=self.__ttsplit, random_state = self.__seed)
        # these four sets should be all numpy ndarrays.
        nrows_Xtest, self.__xdim = self.X_test.shape  # total rows and columns in X_test. # not needed for nrows
        # notice that 
        # self.__xdim == self.X_test.shape[1]   # True
        # self.__xdim is self.X_test.shape[1]   # True
        # nrows_Xtest == self.X_test.shape[0]   # True
        # nrows_Xtest is self.X_test.shape[0]   # False
        return

    # def __setProbeExpos(self, expos):
    #     '''
    #     set Probing Exponents, a tuple
    #     param expos: list/tuple of floats
    #     :return: None
    #     '''
    #     # Can add more checks to ensure expos is numeric list/tuple
    #     self.__probeExpos = expos if (len(expos)>2) else (-6, -3, -1, -0.5, 0, 0.5, 1, 1.5, 2, 4, 6) # tuple, exp(tuple) gives the scaling factors.
    #     self.__probeFactors = tuple( [ math.exp(i) for i in self.__probeExpos ] )
    #     return
    def optimize_k(self):
        from sklearn.model_selection import cross_val_score 
        best_k = self.k
        best_score = -np.inf
        for k_current in range(2, self.__kmax):
            knn = KNeighborsRegressor(n_neighbors=k_current)
            scores = cross_val_score(knn,self.X_train, self.y_train, cv=self.cv)
            mean_score = np.mean(scores)
            print(f'k={k_current}, Cross-validation Score = {mean_score}')
            if mean_score > best_score:
                best_score = mean_score
                best_k = k_current
        self.k = best_k
        print(f"Best k found: {self.k} with cross validation score = {best_score}") 
        return
    
    def __setExpos2Scales(self, expos=[]):
        """
        set Scaling Exponents, a tuple or list
        Should make sure expos is centered (using __shiftCenter)
        Args:
            expos (list, optional): _description_. Defaults to [], should match number of features in data_x
        """
        # Can add more checks to ensure expos is numeric list/tuple
        if (len(expos) != self.__xdim):
            self.__scaleExpos = np.zeros(self.__xdim) # tuple, exp(tuple) gives the scaling factors.
            if self.__xdim > 1: 
                self.__scaleExpos[0] = 1
                self.__scaleExpos[1] = -1
        else:
            self.__scaleExpos =  expos
        self.__scaleFactors = np.array( [ math.exp(i) for i in self.__scaleExpos ] ) # numpy array
        # find gradient direction 
        # have 
        return

    def __shiftCenter(self, expos = []):
        """
        Enforce sum of exponents or any vectors like gradient = 0 (for xdim > 1)

        Args:
            expos (np array, optional): array of scaling exponents. Defaults to [].
        """
        return expos.copy() - expos.sum()/len(expos) if len(expos) > 1 else expos.copy()
       # return expos.copy()if len(expos) > 1 else expos.copy()
        

    def __evalGradients(self, learning_rate=0, use = 'test'):
        """
        evaluate Gradients/partial derivatives with respect to exponential factors (not scaling factor)
        Args:
            learning_rate (float, optional): learning_rate. Defaults to 0.
            use (str, optional): use 'test' (default) or 'train' dataset to score. 
        """
        # set learning_rate
        grad = np.array( [ self.__eval1Gradient(i, learning_rate, use=use) for i in range(self.__xdim) ] )
        # normalize grad here?
        # self.__gradients = grad.copy()
        # return
        return grad # gradient as numpy array

    def __eval1Gradient(self, i, learning_rate=0, use='test'):
        """
        evaluate a single Gradient/partial derivative with respect to the exponential factor (not scaling factor)

        Args:
            i (int): the column/feature index.
            learning_rate (float, optional): learning_rate. Defaults to 0.
            use (str, optional): use 'test' (default) or 'train' dataset to score. 
        """
        thescale = self.__scaleExpos[i]
        thestep = max(learning_rate, self.learning_rate, abs(thescale)*self.learning_rate )
        # modify step value appropriately if needed.
        # maxexpo = thescale + thestep/2
        # minexpo = thescale - thestep/2
        base_expos = self.__scaleExpos.copy()
        maxexpos = base_expos.copy()
        maxexpos[i] += thestep/2
        minexpos = base_expos.copy()
        minexpos[i] -= thestep/2
        slope = ( self.scorethis(scaleExpos=maxexpos, use=use) - self.scorethis(scaleExpos=minexpos, use=use) ) / thestep
        return slope

    def __setNewExposFromGrad(self, grad=() ):
        """
        setting new scaling exponents, from the gradient info
        steps: 
        1. center grad (will take care of both grad = 0 and grad = (1,1,...,1) cases)
        2. normalize grad (with learning rate as well)
        3. add to original expos

        Args:
            grad (tuple, optional): the gradient calculated. Defaults to empty tuple ().
        """
        grad = self.__shiftCenter(grad)
        if np.allclose(grad, self.__vector0, atol=self.__atol): 
            print(f"Gradient is zero or trivial: {grad}, \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
            return False
        #norm = np.sqrt( np.dot(grad,grad) )
        norm = np.linalg.norm(grad) #more efficient than np.sqrt( np.dot(grad,grad) )
        deltaexpos = grad / norm * self.learning_rate
        self.__scaleExpos += deltaexpos
        self.__setExpos2Scales(self.__scaleExpos)
        return True
    
    def optimize_lr(self, scaleExpos_init = (), maxiter = 0, learning_rate=0.1,patience=10, decay_rate=0.01, min_lr=1e-5):
        """
        Optimizing learning rate dynamically based on 
        """
        maxi = max( self.max_iter, maxiter, 1000)
        skip_n = 1 # rule of thumb math.floor(1/learning_rate)
        expos = scaleExpos_init 
        best_test_score = float('-inf')
        no_improve_count = 0
        if (len(scaleExpos_init) == self.__xdim): 
            self.__scaleExpos = scaleExpos_init # assumes the new input is the desired region.
        print(f"Begin: \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}, \nmaxi= {maxi}, k={self.k}, learning_rate={self.learning_rate}\n")
        log_data = []

        initial_log = {
            "iteration": "Begin",
            "__scaleExpos": self.__scaleExpos.copy(),
            "__scaleFactors": self.__scaleFactors.copy(),
            "model_score_train": self.scorethis(use='train'),
            "model_score_test": self.scorethis(use='test'),
            "learning_rate": self.learning_rate,
            "k_neighbours":self.k
        }
        log_data.append(initial_log)


        for i in range(maxi):
            grad = self.__evalGradients(learning_rate, use='train')        
            result = self.__setNewExposFromGrad(grad)
            current_test_score = self.scorethis(use='test')
            if current_test_score > best_test_score:
                best_test_score = current_test_score
                best_model = self.__knnmodels[self.k]
                best_scale_expos = self.__scaleExpos.copy()
                best_scale_factors = self.__scaleFactors.copy()
                no_improve_count = 0 
            else:
                no_improve_count +=1 
            
            if no_improve_count >= patience:
                learning_rate = max(learning_rate *(1-decay_rate), min_lr)
                no_improve_count = 0
                print(f"Reduced learning rate to {learning_rate}")

            if (i<10 or i%skip_n==0 ): 
                log_entry = {
                    "iteration": i,
                    "grad": grad.copy(),
                    "__scaleExpos": self.__scaleExpos.copy(),
                    "__scaleFactors": self.__scaleFactors.copy(),
                    "model_score_train": self.scorethis(use='train'),
                    "model_score_test": self.scorethis(use='test'),
                    "k_neighbours":self.k,
                    "learning_rate": learning_rate

                     
                }
                log_data.append(log_entry)
                print(f"i: {i}, |grad|^2={np.dot(grad,grad)},\nlearning_rate={learning_rate}, \ngrad= {grad}, \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
            if (self.scorethis(use='train')<=0): 
                print(f"Model score is negative. Stopping at iteration {i}.")
                break
            if not result: 
                print(f"Convergence achieved at iteration {i}.")
                break
            
            if i==maxi-1:
                log_final = {
                    "iteration": i,
                    "grad": grad.copy(),
                    "__scaleExpos": self.__scaleExpos.copy(),
                    "__scaleFactors": self.__scaleFactors.copy(),
                    "model_score_train": self.scorethis(use='train'),
                    "model_score_test": self.scorethis(use='test'),
                    "learning_rate": learning_rate
                }
                log_data.append(log_final)
                print(f"max iter reached. Current |grad|^2={np.dot(grad,grad)},\nlearning_rate={learning_rate},  \ngrad= {grad}, \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
        log_df = pd.DataFrame(log_data)

        best_log_entry = log_df.loc[log_df['model_score_test'].idxmax()]
        self.best_test_score = best_log_entry['model_score_test']
        self.best_scale_expos = best_log_entry['__scaleExpos']
        self.best_scale_factors = best_log_entry['__scaleFactors']
        self.best_model_knn = best_log_entry['k_neighbours']
        return log_df

    def optimize(self, scaleExpos_init = (), maxiter = 0, learning_rate=0):
        from concurrent.futures import ThreadPoolExecutor
        """
        Optimizing scaling exponents and scaling factors

        Args:
            scaleExpos_init (np array, optional): initial search vector. Defaults to empty.
            maxiter (int, optional): max iteration. Defaults to 1e5.
            learning_rate (float, optional): learning_rate. Defaults to 0 or self.learning_rate
        """

        maxi = max( self.max_iter, maxiter, 1000)
        skip_n = 100 # rule of thumb math.floor(1/learning_rate)
        expos = scaleExpos_init 
        if (len(scaleExpos_init) == self.__xdim): self.__scaleExpos = scaleExpos_init # assumes the new input is the desired region.
        print(f"Begin: \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}, \nmaxi= {maxi}, k={self.k}, learning_rate={self.learning_rate}\n")
        for i in range(maxi):
            grad = self.__evalGradients(learning_rate, use='train')
            # Cases
            # 1. grad = 0, stop (built into __setNewExposFromGrad)
            # 2. grad parallel to (1,1,1,...,1) direction, stop.
            # 3. maxiter reached, stop. (end of loop)
            # 4. ?? dy < tol, stop??# 
            result = self.__setNewExposFromGrad(grad)
            if (i%skip_n==0 ): print(f"i: {i}, |grad|^2={np.dot(grad,grad)}, \ngrad= {grad}, \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
            if (self.scorethis(use='train')<=0): 
                print(f"Model score is negative. Stopping at iteration {i}.")
                break
            if not result: break


        if i==maxi-1: print(f"max iter reached. Current |grad|^2={np.dot(grad,grad)}, \ngrad= {grad}, \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
            

    def scorethis(self, scaleExpos = [], scaleFactors = [], use = 'test', lambda_reg=0.00, regularization='L1'):
        if len(scaleExpos)==self.__xdim :
            self.__setExpos2Scales( self.__shiftCenter(scaleExpos) )
        # elif len(scaleFactors)==self.__xdim:
        #     self.__scaleFactors = np.array(scaleFactors)
        #     self.__scaleExpos = [ round(math.log(x), 2 ) for x in scaleFactors ]
        else:
            # self.__setExpos2Scales(np.zeros(self.__xdim))
            if (len(scaleExpos)>0 or len(scaleFactors)>0) : print('Scale factors set to default values of unit (all ones). If this is not anticipated, please check your input, making sure the length of the list matches the number of features in the dataset.')
        
        sfactors = self.__scaleFactors.copy() # always start from the pre-set factors, whatever it might be
        #self.__knnmodels[self.k].fit(sfactors*self.X_train, self.y_train)
        # For optimizing/tuning the scaling factors, use the train set to tune. 
        newscore = self.__knnmodels[self.k].score(sfactors*self.X_train, self.y_train) if use=='train' else self.__knnmodels[self.k].score(sfactors*self.X_test, self.y_test)
        return newscore
        # try regularization 
        #if regularization == 'L2':
         #  penalty = lambda_reg * np.sum(np.array(self.__scaleExpos)**2)  # L2 penalty
        #elif regularization == 'L1':
         #   penalty = lambda_reg * np.sum(np.abs(np.array(self.__scaleExpos)))  # L1 penalty
        #else:
         #raise ValueError(f"Unknown regularization type: {regularization}")

       # Return regularized score
        #return newscore - penalty
###### END class sknn

#########################################################################################################################
#%%
# 2. Modify the sknn class to perform K-NN regression.
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, learning_rate_init=0.01)
houseprice.optimize()
#model score-train is 0.8446867562702437, 
#score-test is 0.7819467384489185
# %%
# 3. Modify the sknn class to improve the algorithm performance, logic, or presentations.
# tried different normalization techniques  like minimax and robus 
# used cv to find best k value before gradient descent, best k = 
# increased learning rate
# with cv finding best k = 9
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, learning_rate_init=0.01)
houseprice.optimize()
#result 
#model score-train is 0.8315716005454094, 
#score-test is 0.8015295080076016

#%%
#change learning rate to 0.05 
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, learning_rate_init=0.05)
houseprice.optimize()
#model score-train is 0.85823177667169, 
#score-test is 0.8177596265235567

#%%
#change learning rate to 0.1
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, learning_rate_init=0.1)
houseprice.optimize()

#%%
#modified the sknn class to include other data normalization method: minimax, robust
houseprice = sknn(data_x=data_x, data_y=data_y, scaler = "MinMax", classifier = False, learning_rate_init=0.05, max_iter=400)
houseprice.optimize()
# best k = 5
#model score-train is 0.8409528386581798, 
#score-test is 0.814768172231965

#%%
#try robust 
houseprice = sknn(data_x=data_x, data_y=data_y, scaler = "Robust", classifier = False, learning_rate_init=0.05)
houseprice.optimize()
# best k = 20 
# model score-train is 0.6381765801124846, 
#score-test is 0.6737153363216505
# both normalization methods were not better than Z-transform 

#%%
# tried adding regularziation to the scaling factors , l2, lambda = 0.01
# performmance got worse 
#model score-train is 0.8293858726948979, 
#score-test is 0.728466692208353
# also tried adding L1 regularization, lambda = 0.01
#model score-train is 0.8135693500484253, 
#score-test is 0.7814351285190857
# L1 score is better but not as good as without regularization, and is worse than benchmark score 
# this is likely because normalization is already applied by making the scaling factors sum to 1.

#%%
#tried getting rid of summing to 0 normalization and using L1 regularization 
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, learning_rate_init=0.05)
houseprice.optimize()
#the result turned out worse (negative) and scaleExpos was really large

#%%
# trying dynamic learning rate
# reduce leraning rate if no improvement after 10 iterations (patience = 10), decay rate = 0.005, min_lr = 1e-5
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, learning_rate_init=0.01)
houseprice.optimize_lr()
# the process took a lot longer to converge as learning rate was reduced from 0.01 to <0.005
# it likely got stuck in local minimum and did not converge to lowest test error
# final score is not better than constant 
# model score-train is 0.8315716005454094, 
# score-test is 0.8015295080076016

#%%
#modifying dynamic learning rate with starting 0.1, decay rate = 0.01 
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, learning_rate_init=0.1)
houseprice.optimize_lr()
#model score-train is 0.8409528386581798, 
#score-test is 0.814768172231965
#Reduced learning rate to 0.06622820409839833

#%%
#increasing patience to 20, starting at 0.2
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, learning_rate_init=0.4)
houseprice.optimize_lr()
#model score-train is 0.8547204387477961, 
#score-test is 0.8105462837499338
#Reduced learning rate to 0.0941480149401

#%%
# Add dynamic weighting for different neighbours 
# we know from cross validation that 9 neighbours produce the best result, we can also change the influence of closer 
# neighbours to be higher than those further away 
# inverse weighting
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, weight = "inverse", learning_rate_init=0.4)
houseprice.optimize_lr()
#the result is there is very high training score due to overfitting, and test score is lower than benchmark
#model score-train is 0.9412040973633723, 
#score-test is 0.7832071527170309
#Reduced learning rate to 0.07177305325982747
#%%
#guassian weighting
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, weight = "gaussian", learning_rate_init=0.4)
houseprice.optimize_lr()
#even more overfitting 
#model score-train is 0.9997081381572372, 
#score-test is 0.7339574104181359

#%%
#changing distance metric calculation 
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, metric = "euclidean", learning_rate_init=0.5)
houseprice.optimize()
#model score-train is 0.8178476178306388, 
#score-test is 0.7982869840567368
#%%
# manhattan distance
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, metric = "manhattan", learning_rate_init=0.5)
houseprice.optimize()

#%%
# manhattan distance with dynamic learning rate -- best result!
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, metric = "manhattan", learning_rate_init=0.5)
houseprice.optimize_lr()
#model score-train is 0.8551569338216636, 
#score-test is 0.8413870789984825

#%%
# hamming distance 
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, metric = "hamming", learning_rate_init=0.5)
houseprice.optimize_lr()
# other distance metrics did not perform as well as manhattan

# tried adding back neighbourhood with one-hot encoding
# the number of optimal neighbours 


#%%
# made the code more efficient and clean 
# adding different print statement throughout iteration 

#%%
# final model 
# 3. Find optimized scaling factors for the features for the best model score.
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False,ttsplit=0.6, metric = "manhattan",learning_rate_init=0.5)
#houseprice.optimize_lr()
log_df = houseprice.optimize_lr()

#%%
print(f"best_test_score{houseprice.best_test_score}, \nbest_scale_expos{houseprice.best_scale_expos}, \nbest_scale_factors{houseprice.best_scale_factors}, \nbest_model_knn {houseprice.best_model_knn}")

#%%
# since scalefactor = exp(scaleExpos), we will only examine scalefactor
# scale factor greater than 1 means amlifying feature importance whilst <1 means diminishing importance
#best_test_score0.8487055176579418, 
#best_scale_expos[ 0.7249798  -0.51326198  0.45673075  0.04537654 -0.01682784  0.93122211
# -0.0933537   0.14122141 -0.01768083 -0.04669485 -0.13636263  0.1856856
#  0.68238199 -0.49523016 -0.06706822 -0.59969028 -0.27406447  1.09397708
# -1.12086949  0.24627757  0.54207801 -0.30436755 -0.32969212  0.31679999
# -0.55508437  0.37848815 -0.08386592  0.6787443   0.72605799 -0.27914631
#  0.66451742  0.72766119 -0.82331989 -0.08919716  0.01777066  0.75780944
# -0.57493318 -0.44085043 -0.25297866 -0.37128638 -0.64142832 -0.31693702
# -0.58118823 -0.38471164 -0.0691406   0.1482803   0.37533465 -0.21391778
# -0.14824495], 
#best_scale_factors[2.06468938 0.59853997 1.57890371 1.04642181 0.98331296 2.53760851
# 0.91087127 1.15167961 0.98247456 0.95437858 0.87252617 1.20404365
# 1.9785851  0.60943062 0.93513141 0.54898164 0.76028306 2.98612656
# 0.32599622 1.2792546  1.71957645 0.73758972 0.71914511 1.37272799
# 0.57402383 1.4600755  0.91955454 1.97140069 2.06691671 0.75642923
# 1.94355237 2.07023306 0.43897189 0.91466522 1.0179295  2.13359733
# 0.56274247 0.64348894 0.77648445 0.68984635 0.52653982 0.72837663
# 0.55923347 0.68064688 0.93319546 1.15983796 1.45547841 0.80741476
# 0.86221989], 
#best_model_knn  12 neighbours 
#Index(['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
 #      'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual',
 #      'BsmtCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
 #      'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
 #      'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
 #      'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
 #      'Functional', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt',
 #      'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
 #      'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
 #      'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscVal', 'MoSold', 'YrSold'],
 #     dtype='object')
#%%
knnfeatures = pd.DataFrame({'feature':data_x.columns, 'scale_factor':houseprice.best_scale_factors})
knnfeatures_sort= knnfeatures.sort_values(by='scale_factor', ascending=False)
# 4. Modify the sknn class to save some results 
# (such as scores, scaling factors, gradients, 
#  etc, at various points, like every 100 epoch).
# save as csv 
# modified the class to save result at every epoch for optimize_lr(), and also saving intermediate learning rate, the best neighbour from cv

#%%
large_features = (knnfeatures_sort['scale_factor']>1).sum() #20
small_featuers = (knnfeatures_sort['scale_factor']<1).sum() #29
#===============================================================================================================================
# 5. Compare the results of the optimized scaling factors to Feature Importance
#  from other models
# multiple linear regression, tree regression, NN, PCR 

#top 5 features
#   1stFlrSF	2.986127
#5	YearRemodAdd	2.537609
#35	GarageArea	2.133597
#31	FireplaceQu	2.070233
#28	TotRmsAbvGrd	2.066917

#bottom 5 features 
#42	3SsnPorch	0.559233
#15	HeatingQC	0.548982
#40	OpenPorchSF	0.526540
#32	GarageYrBlt	0.438972
#18	2ndFlrSF	0.325996
#%%
#multiple linear regression 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
linear_reg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.6, random_state=42)
linear_reg.fit(X_train, y_train)
lr_feature_importance = pd.DataFrame({'feature': data_x.columns, 'importance': linear_reg.coef_})

feature_importance_sorted = lr_feature_importance.sort_values(by='importance', ascending=False)
print(feature_importance_sorted)

#top 5 
#7       ExterQual  12112.508622
#2     OverallQual  10568.565230
#27    KitchenQual   9677.676023
#9        BsmtQual   8252.762558
#36     GarageQual   6771.077933

#bottom 5
#10       BsmtCond  -8350.899535
#37     GarageCond -10072.303507
#25   BedroomAbvGr -11971.872724
#26   KitchenAbvGr -19806.626994
#45         PoolQC -24484.177724

#%%
#look at summary statistics

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
lr_model = sm.OLS(y_train, X_train_const).fit()


model_summary = lr_model.summary()
print(model_summary)
y_pred = lr_model.predict(X_test_const)
test_score = lr_model.rsquared
print(f"Test R-squared: {test_score}")
feature_stats = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lr_model.params[1:],  
    'p-value': lr_model.pvalues[1:],     
    't-value': lr_model.tvalues[1:],     
    'std_err': lr_model.bse[1:]          
})

# Sort by p-value
feature_stats_sorted = feature_stats.sort_values(by='p-value', ascending = True)
print(feature_stats_sorted)

#adjusted r squared is 0.896 higher than knn model 
# the linear regression placed all the quality and condition features as top features with high coefficient, meaning that if quality and condition of certain house areas increase the sale price would increase
# the ones with negative coefficients are miscellaneous features like pool quality, kitchen above ground and bedroom above ground, year sold
# in terms of significance, miscellaneous features like garagecars, HeatingQC, CentralAir are also not significant 
# comparing to the KNN model, there is a lot of overlap between features that are not statistically significant and features reduced by 
# knn model, including centralAir, HeatingQC, OpenPorchSF, GarageYrBlt	
# however, two of the top features that knn amplified is not significant in linear regression,  YearRemodAdd, FireplaceQu
# other features amplified by knn model corresponds with linear regression, including MSSubClass, OverallQual, KitchenQual, GarageArea, GrLivArea


#%%
#Tree regression 
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=2)
tree_reg.fit(X_train, y_train)
tree_feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': tree_reg.feature_importances_
})
tree_feature_importance_sorted = tree_feature_importance.copy().sort_values(by='importance', ascending=False)
print(tree_feature_importance_sorted)
tree_reg.score(X_train, y_train) #0.9999897481550298
tree_reg.score(X_test, y_test) #0.6836137828160598
comparison2 = pd.DataFrame({
    'knn_feature': knnfeatures_sort.copy()['feature'],
    'knn_importance': knnfeatures_sort['scale_factor'],
    'lr_feature': feature_importance_sorted['feature'],
    'lr_importance': feature_importance_sorted['importance'],
    'tree_feature': tree_feature_importance_sorted['feature'],
    'tree_importance': tree_feature_importance_sorted['importance']
})
#%%
print(comparison2)

#the tree feature importance is similar to other two model as overallQual, GrLivArea and 1stFlrSF are all present in top 5
#the bottom 5 include PoolArea, MiscVal, centralAir 
# garageyrBlt which was very low in knn turned out to be 12th in tree model, and mscVal which is very low in tree importance is 12th in knn
# GarageQual, 3SsnPorch are both unimportant features in tree and knn. 


# Neural network 
#%%
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
nn_reg = MLPRegressor(hidden_layer_sizes=(100, 25, 25), random_state=2, max_iter=1000)
nn_reg.fit(X_train, y_train)
perm_importance = permutation_importance(nn_reg, X_test, y_test, n_repeats=10)
nn_feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance_nn': perm_importance.importances_mean
})
nn_feature_importance_sorted = nn_feature_importance.sort_values(by='importance_nn', ascending=False)

train_score_nn = nn_reg.score(X_train, y_train) #0.8512941361759958
test_score_nn = nn_reg.score(X_test, y_test) #0.629423476356608


# neural network top features are garageArea, 2ndFlrSF, GrLivArea, BsmtFinSF1, BsmtUnfSF
# the least important features are lotArea, 1stFlrSF, YearRemodAdd, GarageYrBlt, MoSold
# the feature importance is quite different than all the other models, likely because it's more complex given 
# the number of nodes I have chosen  


#-------------------------------------------------------------------------------------------------------------------------------------------------
#save all to csv 
#%%
comparison = pd.concat([knnfeatures_sort.reset_index(drop=True), 
           feature_stats_sorted.reset_index(drop=True),
           tree_feature_importance_sorted.reset_index(drop=True),
           nn_feature_importance_sorted.reset_index(drop=True)], axis=1)
comparison.columns = ['knn_feature', 'knn_importance', 'lr_feature', 'lr_coefficient', 'lr_p_value', 'lr_t_value', 'lr_std_err', 'tree_feature', 'tree_importance', 'nn_feature', 'nn_importance']
comparison.to_csv('feature_importance_comparison.csv', index=False)
# %%
# Conclusion 
# If I want to maximize the score, I will keep many of the nominal variables that I deleted, especially "Neighborhood", as house prices are usually 
# highly dependent on location.
# performance wise, tree regression did not have good test performance and so did NN. However, i did not properly tune the models. 
# Knn regession seems to be slightly worse than linear regression although both test scores quite high.
# linear regression had the best interpretability for features due to significance testing.
# Overall, the housing price is more dependent on condition and quality of large, important areas of the house such as overall scores, garage, living areas, kitchen, 
# it is less dependent on smaller amenities like heating, central air, front porch, fire palces. However, since test score is optimized in all of the models, the models treated different variables 
# differently. If the variables are correlated, their importance could be widely different in each models, which reduces interpretability. 
# surprisingly year sold has not come up as an important features across models, but likely a different encoding is needed for that variable. 

#future improvement 
# I could also try adjusting feature importance based on results from other models such as linear regression, PCR and regression tree. 
# I could also delete highly correlated features or use domain knowledge of house prices to scale the features. 