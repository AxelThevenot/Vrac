import numpy as np
import pandas as pd

class Scaler:
    
    def __init__(self):
        self.scales = {}         
            
    def log(self, df, col_name, forward=True, is_pos=True):
        
        # assume the data is on [0, +inf[
        eps = 1

        if forward:
            maxi = df[col_name].max()
            self.scales[col_name]['log'] = {"maxi": maxi}

            if not is_pos:
                # invert the distribution look like
                df[col_name] = maxi - df[col_name]

            # apply log transform
            df[col_name] = np.log(df[col_name] + eps)

            
        else:
            maxi = self.scales[col_name]['log']['maxi']
            # remove the log transform
            df[col_name] = np.exp(df[col_name]) - eps
            if not is_pos:
                # invert the distribution look like
                df[col_name] = maxi - df[col_name]
    
        
    def standardize(self, df, col_name, forward=True, **kwargs):
        if forward:
            mean = df[col_name].mean()
            std = df[col_name].std()
            df[col_name] = (df[col_name] - mean) / std
            return {"mean": mean, "std": std}
        else:
            df[col_name] = df[col_name] * kwargs["std"] + kwargs["mean"]
        
    def normalize(self, df, col_name, forward=True, **kwargs):
        if forward:
            mini = df[col_name].min()
            maxi = df[col_name].max()
            df[col_name] = (df[col_name] - mini) / (maxi - mini)
            return {"mini": mini, "maxi": maxi}
        else:
            df[col_name] = df[col_name] * (kwargs["maxi"] - kwargs["mini"]) + kwargs["mini"]
            
    def fit(self, df, col_name, t='standardize', pre_t=None):
        self.fit_transform(df, col_name, t, pre_t, transform=False)
        
    def transform(self, df, col_name, t='standardize', pre_t=None):
        self.fit_transform(df, col_name, fit=False)

    def transform_all_df(self, df):
        for col_fitted in [col for col in self.scales if col in df.columns]:
            self.fit_transform(df, col_fitted, fit=False)
          
    
    def fit_transform(self, df, col_name, t='standardize', pre_t=None, fit=True, transform=True):
        # if not tranform (only fit) then copy the datatframe to not change it
        if not transform:
            df = df.copy()
        
        # if not fit (only transform) then assume the fitting exists 
        t_kwargs = {}
        if not fit:
            t = self.scales[col_name]['t']
            pre_t = self.scales[col_name]['pre_t']
            t_kwargs = self.scales[col_name]['t_kwargs']
        else:
            self.scales[col_name] = {}
        
        # pretransformation
        if pre_t == 'pos_log':
            self.log(df, col_name)
        elif pre_t == 'neg_log':
            self.log(df, col_name, is_pos=False)
        elif pre_t is None:
            pass
        else:
            print(f'Pre-transformation {pre_t} not found')
            
        # transformation   
        if t == 'normalize':
            t_kwargs = self.normalize(df, col_name, **t_kwargs)
        elif t == 'standardize':
            t_kwargs = self.standardize(df, col_name, **t_kwargs)
        else: 
            print(f'Transformation {t} not found')
        
        # save the fit 
        if fit:
            self.scales[col_name]['t'] = t 
            self.scales[col_name]['pre_t'] = pre_t 
            self.scales[col_name]['t_kwargs'] = t_kwargs 

    # inverse of transformations
    def untransform(self, df, col_name, like_col_name=None):
        if like_col_name is not None:
            self.scales[col_name] = self.scales[like_col_name]
        
        _dict = self.scales[col_name]
        if _dict["t"] == 'normalize':
            self.normalize(df, col_name, forward=False, **_dict["t_kwargs"])
        elif _dict["t"] == 'standardize':
            self.standardize(df, col_name, forward=False, **_dict["t_kwargs"])
        else:
            pass

            
        # pretransformation
        if _dict["pre_t"] == 'pos_log':
            self.log(df, col_name, forward=False)
        elif _dict["pre_t"] == 'neg_log':
            self.log(df, col_name, forward=False, is_pos=False)
            
    
