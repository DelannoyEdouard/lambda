# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 09:35:48 2022

@author: edoua
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#importation du jeu de donné nettoyé
df=pd.read_csv('predClean.csv',sep=';',index_col=0)
pd.set_option('display.max_column', 81)
print(df.shape)
print(df.head())

#valeur manquante ordonnée
nulls=pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns=['Null Count']
nulls.index.name='Feature'
#nulls
print(nulls)

#suppresion de la colonne Identifiant dans data1
del df['Id']

#changement des valeurs objets en numérique pour ne pas dummiser ou mettre des valeurs aléatoirements selon leur répartition
df['MSZoning']=df['MSZoning'].replace(to_replace={'RL':0,'RM':1,'FV':2,'RH':3,'C (all)':4})
df[['Street','Alley']]=df[['Street','Alley']].replace(to_replace={'Pave':1,'Grvl':2}).astype(int)
df['LotShape']=df['LotShape'].replace(to_replace={'Reg':0,'IR1':1,'IR2':2,'IR3':3})
df['LandContour']=df['LandContour'].replace(to_replace={'Lvl':0,'HLS':1,'Bnk':2,'Low':3})
df['Utilities']=df['Utilities'].replace(to_replace={'AllPub':0,'NoSeWa':1})
df['LotConfig']=df['LotConfig'].replace(to_replace={'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4})
df['LandSlope']=df['LandSlope'].replace(to_replace={'Gtl':0,'Mod':1,'Sev':2})
df['Neighborhood']=df['Neighborhood'].replace(to_replace={'NAmes':0,'CollgCr':1,'OldTown':2,'Edwards':3,'Somerst':4,'NridgHt':5,
                                                          'Gilbert':6,'SawyerW':7,'Mitchel':8,'BrkSide':9,'Crawfor':10,'IDOTRR':11,
                                                          'Timber':12,'NoRidge':13,'StoneBr':14,'SWISU':15,'ClearCr':16,'MeadowV':17,
                                                          'BrDale':18,'Blmngtn':19,'Veenker':20,'NPkVill':21,'Blueste':22,
                                                          'NWAmes':23,'Sawyer':24})
df[['Condition1','Condition2']]=df[['Condition1','Condition2']].replace(to_replace={'Norm':0,'Feedr':1,'Artery':2,'RRAn':3,
                                                                                    'PosN':4,'RRAe':5,'PosA':6,'RRNn':7,'RRNe':8})
df['BldgType']=df['BldgType'].replace(to_replace={'1Fam':0,'TwnhsE':1,'Duplex':2,'Twnhs':3,'2fmCon':4})
df['HouseStyle']=df['HouseStyle'].replace(to_replace={'SLvl':0,'1Story':1,'SFoyer':2,'1.5Unf':3,'1.5Fin':4,'2Story':5,'2.5Unf':6,
                                                      '2.5Fin':7})
df['RoofStyle']=df['RoofStyle'].replace(to_replace={'Gable':0,'Hip':1,'Gambrel':2,'Flat':3,'Mansard':4,'Shed':5})
df['RoofMatl']=df['RoofMatl'].replace(to_replace={'CompShg':0,'Tar&Grv':1,'WdShake':2,'WdShngl':3,'ClyTile':4,'Membran':5,
                                                  'Metal':6,'Roll':7})
df[['Exterior1st','Exterior2nd']]=df[['Exterior1st','Exterior2nd']].replace(to_replace={'VinylSd':0,'MetalSd':1,'HdBoard':2,
                                                                   'Wd Sdng':3,'Plywood':4,'CemntBd':5,'WdShing':6,'Stucco':7,
                                                                   'BrkFace':8,'AsbShng':9,'BrkComm':10,'ImStucc':11,'Stone':12,
                                                                   'AsphShn':13,'CBlock':14,'Other':15})
df['Exterior2nd']=df['Exterior2nd'].replace(to_replace={'CmentBd':5,'Wd Shng':6,'Brk Cmn':10}).astype(int)
df['MasVnrType']=df['MasVnrType'].replace(to_replace={'None':0,'BrkFace':1,'Stone':2,'BrkCmn':3}).astype(int)
df[['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']]=df[['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']].replace(to_replace={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}).astype(int)

#changement des valeurs objets en numérique pour ne pas dummiser ou mettre des valeurs aléatoirements selon leur répartition.Encore
df['Foundation']=df['Foundation'].replace(to_replace={'PConc':0,'CBlock':1,'BrkTil':2,'Slab':3,'Wood':4,'Stone':5})
df['BsmtExposure']=df['BsmtExposure'].replace(to_replace={'No':1,'Mn':2,'Av':3,'Gd':4}).astype(int)
df[['BsmtFinType1','BsmtFinType2']]=df[['BsmtFinType1','BsmtFinType2']].replace(to_replace={'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,
                                                                                            'ALQ':5,'GLQ':6}).astype(int)
df['Heating']=df['Heating'].replace(to_replace={'GasA':0,'GasW':1,'Grav':2,'Wall':3,'OthW':4,'Floor':5})
df['CentralAir']=df['CentralAir'].replace(to_replace={'N':0,'Y':1})
df['Electrical']=df['Electrical'].replace(to_replace={'SBrkr':0,'FuseA':1,'FuseF':2,'FuseP':3,'Mix':4})
df['Functional']=df['Functional'].replace(to_replace={'Typ':0,'Min2':1,'Min1':2,'Mod':3,'Maj1':4,'Maj2':5,'Sev':6})
df['GarageType']=df['GarageType'].replace(to_replace={'Attchd':1,'Detchd':2,'BuiltIn':3,'Basment':4,'2Types':5,'CarPort':6}).astype(int)
df['GarageFinish']=df['GarageFinish'].replace(to_replace={'Unf':1,'RFn':2,'Fin':3}).astype(int)
df['PavedDrive']=df['PavedDrive'].replace(to_replace={'N':1,'P':2,'Y':3}).astype(int)
df['Fence']=df['Fence'].replace(to_replace={'MnPrv':1,'MnWw':2,'GdWo':3,'GdPrv':4}).astype(int)
df['MiscFeature']=df['MiscFeature'].replace(to_replace={'Shed':1,'Gar2':2,'Othr':3,'TenC':4}).astype(int)
df['SaleType']=df['SaleType'].replace(to_replace={'WD':0,'New':1,'COD':2,'ConLD':3,'CWD':4,'ConLI':5,'ConLw':6,'Oth':7,'Con':8})
df['SaleCondition']=df['SaleCondition'].replace(to_replace={'Normal':0,'Partial':1,'Abnorml':2,'Family':3,'Alloca':4,'AdjLand':5})

#il ne reste plus de colonne objet
print(df.info())

#Changement de la colonne année de construction,travaux, construction garage et vente par age du batîment,renovation, garage 
#et vente
df['AgeMaison']=2010-df['YearBuilt']
df['DrTr']=2010-df['YearRemodAdd']
df['CnGa']=2010-df['GarageYrBlt']
df['Vente']=2010-df['YrSold']
#Suppression des anciennes colonnes
df=df.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'],axis=1) 
#on change les 147 valeurs manquantes dans la colonne CnGa par -1
df['CnGa'].fillna(-1, inplace=True)

#valeur manquante ordonnée
nulls=pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns=['Null Count']
nulls.index.name='Feature'
#nulls
print(nulls)
print(df.head())

#Stockage du jeu de donnée clean et preparer
df.to_csv('predPrpo.csv', sep=';')