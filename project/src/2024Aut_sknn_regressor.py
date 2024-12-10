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
'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
'Condition1',	'Condition2',	'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
'Exterior1st',	'Exterior2nd','MasVnrType',
'Foundation',	'BsmtExposure',	'BsmtFinType1',
'BsmtFinType2', 'Heating',	 'Electrical',
'GarageType',  'Fence', 'MiscFeature', 'SaleType',  'SaleCondition'
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
df.shape[1]
data_x = df.iloc[:, 0:50] # data_x is pandas df here, 
data_y = df.iloc[:, 51] 

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
                    ttsplit=0.5, 
                    max_iter = 100, 
                    seed=1, 
                    scoredigits = 6, 
                    learning_rate_init = 0.5, 
                    atol = 1e-8, 
                    scaler = 'Ztransform' ,
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
        for i in range(2,self.__kmax +1): 
            if (self.__classifierTF): 
                self.__knnmodels.append( KNeighborsClassifier(n_neighbors=i, weights='uniform').fit(self.X_train, self.y_train ) )
            else: 
                self.__knnmodels.append( KNeighborsRegressor(n_neighbors=i, weights='uniform').fit(self.X_train, self.y_train ) ) # TODO
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
        thestep = max(learning_rate, self.learning_rate, abs(thescale)*self.learning_rate ) # modify step value appropriately if needed.
        # maxexpo = thescale + thestep/2
        # minexpo = thescale - thestep/2
        maxexpos = self.__scaleExpos.copy()
        maxexpos[i] += thestep/2
        minexpos = self.__scaleExpos.copy()
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
    
    def optimize_lr(self, scaleExpos_init = (), maxiter = 0, learning_rate=0.01,patience=10, decay_rate=0.005, min_lr=1e-5):
        """
        Optimizing learning rate dynamically based on 
        """
        maxi = max( self.max_iter, maxiter, 1000)
        skip_n = 50 # rule of thumb math.floor(1/learning_rate)
        expos = scaleExpos_init 
        best_test_score = float('inf')
        no_improve_count = 0
        if (len(scaleExpos_init) == self.__xdim): self.__scaleExpos = scaleExpos_init # assumes the new input is the desired region.
        print(f"Begin: \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}, \nmaxi= {maxi}, k={self.k}, learning_rate={self.learning_rate}\n")
        for i in range(maxi):
            grad = self.__evalGradients(learning_rate, use='train')        
            result = self.__setNewExposFromGrad(grad)
            current_test_score = self.scorethis(use='test')
            if current_test_score < best_test_score:
                best_test_score = current_test_score
                no_improve_count = 0 
            else:
                no_improve_count +=1 
            
            if no_improve_count >= patience:
                learning_rate = max(learning_rate *(1-decay_rate), min_lr)
                no_improve_count = 0
                print(f"Reduced learning rate to {learning_rate}")

            if (i<10 or i%skip_n==0 ): print(f"i: {i}, |grad|^2={np.dot(grad,grad)}, \ngrad= {grad}, \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
            if (self.scorethis(use='train')<=0): 
                print(f"Model score is negative. Stopping at iteration {i}.")
                break
            if not result: 
                print(f"Convergence achieved at iteration {i}.")
                break
            
        if i==maxi-1:
            print(f"max iter reached. Current |grad|^2={np.dot(grad,grad)}, \ngrad= {grad}, \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
            

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
        #if regularization == 'L2':
         #  penalty = lambda_reg * np.sum(np.array(self.__scaleExpos)**2)  # L2 penalty
        #elif regularization == 'L1':
         #   penalty = lambda_reg * np.sum(np.abs(np.array(self.__scaleExpos)))  # L1 penalty
        #else:
         #raise ValueError(f"Unknown regularization type: {regularization}")

       # Return regularized score
        #return newscore - penalty
###### END class sknn

#%%
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
#adding regularziation to the scaling factors , l2, lambda = 0.01
# performmance got worse 
#model score-train is 0.8293858726948979, 
#score-test is 0.728466692208353
# adding L1 regularization, lambda = 0.01
#model score-train is 0.8135693500484253, 
#score-test is 0.7814351285190857
# L1 score is better but not as good as without regularization, and is worse than benchmark score 
# this is likely because normalization is already applied by making the scaling factors sum to 1.

#%%
#got rid of summing to 0 normalization and using L1 regularization 
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, learning_rate_init=0.05)
houseprice.optimize()
#the result turned out worse (negative) and scaleExpos was really large

#%%
#dynamic learning rate
# reduce leraning rate if no improvement after 10 iterations (patience = 10), decay rate = 0.005, min_lr = 1e-5
houseprice = sknn(data_x=data_x, data_y=data_y, classifier = False, learning_rate_init=0.01)
houseprice.optimize_lr()
# the process took a lot longer to converge as learning rate was reduced from 0.01 to <0.007






#making the code more efficient 



#%%
# final model 
# 3. Find optimized scaling factors for the features for the best model score.






# 4. Modify the sknn class to save some results 
# (such as scores, scaling factors, gradients, 
#  etc, at various points, like every 100 epoch).
# modified the class to save result at every 100th epoch, and 


# 5. Compare the results of the optimized scaling factors to Feature Importance
#  from other models
# multiple linear regression, tree regression, NN, PCR 
# 
# %%
# Conclusion 
# If I want to maximize the score, I will keep many of the nominal variables that I deleted, especially "Neighborhood", as house prices are usually 
# highly dependent on location.