# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 09:09:02 2022

@author: edoua
"""

"""Projet Kaggle - Nettoyage du jeu de donnée sur les ventes immobilières à Ames en Iowa"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Importation des jeux de données train et test en vue de les fusionnées
dfTrain=pd.read_csv('train.csv',sep=',')
pd.set_option('display.max_column', 81)
print(dfTrain.info())

dfTest=pd.read_csv('test.csv',sep=',')
print(dfTest.info())
print(dfTest.head())

dfTestTarget=pd.read_csv('sample_submission.csv',sep=',')
print(dfTestTarget.info())
print(dfTestTarget.head())

dfTestC=dfTest.merge(dfTestTarget, how='inner', on='Id')
print(dfTestC.head())

df=pd.concat([dfTrain, dfTestC], axis=0)
print(df.info())

#valeur manquante ordonnée
nulls=pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns=['Null Count']
nulls.index.name='Feature'
#nulls
print(nulls)

#Recherche de valeurs incohérantes pour PoolQC en vue de changer les np.nan par 0 si il n'y a pas de piscine.
piscine=df[(df['PoolArea']!=0)&(df['PoolQC'].isna())]
print(piscine.shape)
print(piscine.head())

#Comparaison en fonction des autres biens ayant une piscine et le prix, sinon mod.
print(df[df['PoolArea']!=0])

#Aux vues des informations le prix comme MSSubClass ne peuvent nous faire tendre vers une valeur. On mettra le mod.
df.iloc[2420,[72]]='Gd'
df.iloc[2503,[72]]='Gd'
df.iloc[2599,[72]]='Gd'
#Pour les autres valeurs manquantes, on partira du principe qu'il n'y a pas de piscine et donc 0 en vue des les numériser par 
#la suite.
df['PoolQC'].fillna('0',inplace=True)

#Recherche de valeur incohérante pour MiscFeature en vue de changer les np.nan par 0, si il n'y a pas fonctionalité diverse
sousEnsemble=df[(df['MiscVal']!=0)&(df['MiscFeature'].isna())]
print(sousEnsemble.shape)
print(sousEnsemble.head())

#Comme la valeur et le max on peut en déduire qu'elle doit être similaire au valeur proche
print(df[df['MiscVal']>10000])

df.iloc[2549,[74]]='Gar2' 
#On change le reste des valeurs manquantes par 0 en vue de la numérisation future de cette colonne
df['MiscFeature'].fillna('0',inplace=True)

#pour la colonne Alley, aucune information peut nous aiguiller donc on partira du principe que les np.nan => pas d'allée
#Comme pour Fence qui informe sur la présence de cloture
df['Alley'].fillna('0', inplace=True)
df['Fence'].fillna('0', inplace=True)

#Recherche de valeur incohérante pour FirePlaceQu en vue de changer les np.nan par 0, si il n'y a pas de cheminée
cheminee=df[(df['Fireplaces']!=0)&(df['FireplaceQu'].isna())]
print(cheminee.shape)

#On remplace les valeurs manquantes pas 0 en vue de la numériser dans un 2nd temps.
df['FireplaceQu'].fillna('0', inplace=True)

#Visualisation de la distance moyenne entre l'habitation et l'exterieure de la propriétée en fonction du quartier
sns.barplot(data=df,x='Neighborhood',y='LotFrontage',estimator=np.median)
plt.xticks(rotation=90)
#On change les valeurs manquantes par la médiane en fonction du quartier
df["LotFrontage"]=df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

#On contaste qu'il y a 157 biens avérés sans garage.
garage=df[df['GarageArea']==0]
print(garage.shape)
#recherche des valeurs incohérantes en vue de les changer
garageV=df[(df['GarageArea']!=0)&(df['GarageQual'].isna())]
print(garageV.shape)
print(garageV.head())

#Création d'un sous ensemble en vue d'approximer au mieux c'est ligne
sousEnsemble1=df[(df['MSZoning']=='RM')&(df['MSSubClass']==70)&(df['Neighborhood']=='IDOTRR')]
print(sousEnsemble1.shape)
print(sousEnsemble1.head(8))
#Dans ce cas le sous ensemble n'est pas assez large et diverge pour avoir une approximation plus fine. 
#On choisira le mode ou la médiane.
df.iloc[2126,63:65]='TA' 
df.iloc[2126,[59]]=1979
df.iloc[2126,[60]]='Unf'

df.iloc[2576,63:65]='TA' 
df.iloc[2576,[59]]=1979
df.iloc[2576,[60]]='Unf'
df.iloc[2576,[61]]=2
df.iloc[2576,[62]]=480

#On fini en modifiant les autres valeurs en partant du principe qu'ils n'ont pas de garage
for col in ('GarageType','GarageFinish','GarageQual','GarageCond'):
    df[col].fillna('0',inplace=True)
for col in ('GarageArea','GarageCars'):
    df[col].fillna(0, inplace=True)
    
#On contaste qu'il y a 78 biens avéré sans sous-sol.
SousSol=df[df['TotalBsmtSF']==0]
print(SousSol.shape)
#On recherche les valeurs incohérantes
sousEns2=df[(df['TotalBsmtSF']!=0)&(df['BsmtCond'].isna())]
print(sousEns2.shape)
print(sousEns2.head())

#Recherche de sous ensemble approximant la valeur rechercher
ensemble1=df[(df['Condition1']=='Norm')&(df['MSZoning']=='RL')&(df['Condition2']=='Norm')&(df['Neighborhood']=='CollgCr')&
             (df['MSSubClass']==80)]
print(ensemble1.shape)
print(ensemble1.head(6))

#Suite à une approximation des valeurs manquantes, on peut prédire une valeur réaliste
df.iloc[2040,[31]]='Gd' 
df.iloc[2120,30:39]='0' 
df.iloc[2185,[31]]='TA'
df.iloc[2524,[31]]='TA' 

#On recherche les valeurs incohérantes
sousEns3=df[(df['TotalBsmtSF']!=0)&(df['BsmtExposure'].isna())]
print(sousEns3.shape)
print(sousEns3.head())

#Recherche de sous ensemble approximant la valeur rechercher
ensemble2=df[(df['MSZoning']=='FV')&(df['Neighborhood']=='Somerst')&(df['MSSubClass']==60)&(df['LotConfig']=='Corner')]
print(ensemble2.shape)
print(ensemble2.head(7))

#Suite à une approximation des valeurs manquantes, on peut prédire une valeur réaliste
df.iloc[948,[32]]='No' 
df.iloc[1487,[32]]='Av'
df.iloc[2348,[32]]='No'

#recherche les valeurs incohérantes
sousEns4=df[(df['TotalBsmtSF']!=0)&(df['BsmtQual'].isna())]
print(sousEns4.shape)
print(sousEns4.head())

#Recherche de sous ensemble approximant la valeur rechercher
ensemble3=df[(df['MSZoning']=='C (all)')&(df['Neighborhood']=='IDOTRR')&(df['MSSubClass']==50)]
print(ensemble3.shape)
print(ensemble3.head(7))

#Suite à une approximation des valeurs manquantes, on peut prédire une valeur réaliste
df.iloc[2217,[30]]='Fa'
df.iloc[2218,[30]]='Fa'

#recherche les valeurs incohérantes
sousEns5=df[(df['TotalBsmtSF']!=0)&(df['BsmtFinType2'].isna())]
print(sousEns5.shape)
print(sousEns5.head())

#Recherche de sous ensemble approximant la valeur rechercher
ensemble4=df[(df['MSZoning']=='RL')&(df['Neighborhood']=='NridgHt')&(df['MSSubClass']==20)&(df['LandContour']=='Lvl')&
            (df['LotShape']=='IR1')&(df['LotConfig']=='Inside')&(df['OverallQual']==8)&(df['OverallCond']==5)]
print(ensemble4.shape)
print(ensemble4.head(8))

#Recherche d'un 2nd sous ensemble approximant la valeur rechercher
test=df[(df['BsmtFinSF2']!=0)&(df['MSZoning']=='RL')&(df['Neighborhood']=='NridgHt')&(df['MSSubClass']==20)]
print(test.shape)
test.head()

#Suite à une approximation des valeurs manquantes, on peut prédire une valeur réaliste
df.iloc[332,[35]]='BLQ'
#Changement des valeurs manquantes restantes en partant du principe qu'elles ont pas de sous-sol
for col in ('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
    df[col].fillna('0',inplace=True)
for col in ('BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'):
    df[col].fillna(0,inplace=True)
    
#On contaste qu'il y a 1738 biens avéré sans placage.
plaquage=df[df['MasVnrArea']==0]
print(plaquage.shape)
#On recherche la valeur manquante
test=df[(df['MasVnrArea']!=0)&(df['MasVnrType'].isna())]
print(test[test['MasVnrArea']>0])

#Recherche de sous ensemble approximant la valeur rechercher
sousEnsemble6=df[(df['MSSubClass']==20)&(df['MSZoning']=='RL')&(df['Neighborhood']=='Mitchel')&(df['Condition1']=='Norm')&
                 (df['Condition2']=='Norm')&(df['LotConfig']=='Inside')&(df['LotShape']=='Reg')&(df['LandContour']=='Lvl')]
print(sousEnsemble6.shape)
print(sousEnsemble6.head(22))

#Suite à une approximation de la valeur manquante, on peut prédire une valeur réaliste
df.iloc[2610,[25]]='BrkFace'
#Changement des valeurs manquantes restantes en partant du principe qu'elles ont pas de placage
df["MasVnrType"].fillna("0",inplace=True)
df["MasVnrArea"].fillna(0,inplace=True)

#Recherche des valeurs manquantes pour MSZoning
print(df[df['MSZoning'].isna()])
#Recherche de sous ensemble approximant la valeur rechercher
sousEnsemble7=df[(df['MSSubClass']==20)&(df['Neighborhood']=='Mitchel')&(df['Condition1']=='Artery')&(df['Condition2']=='Norm')]
print(sousEnsemble7.shape)
print(sousEnsemble7.head())

#Suite à une approximation de la valeur manquante, on peut prédire une valeur réaliste
df.iloc[1915,[2]]='C (all)'
df.iloc[2216,[2]]='C (all)'
df.iloc[2250,[2]]='RM'
df.iloc[2904,[2]]='RL'

##Recherche des valeurs manquantes pour Functional
print(df[df['Functional'].isna()])

#Recherche de sous ensemble approximant la valeur rechercher
sousEnsemble8=df[(df['MSSubClass']==50)&(df['MSZoning']=='RM')&(df['Neighborhood']=='IDOTRR')&(df['Condition1']=='Artery')&(df['Condition2']=='Norm')]
print(sousEnsemble8.shape)
print(sousEnsemble8.head())
#Suite à une approximation de la valeur manquante, on peut prédire une valeur réaliste
df.iloc[2216,[55]]='Typ'
df.iloc[2473,[55]]='Typ'

#Vue qu'il y a 2 valeurs possibles et que la 2ème valeur a une occurence sur 2919, on choisit le mod
df["Utilities"].fillna("AllPub",inplace=True)
#Recherche des valeurs manquantes pour Electrical
print(df[df['Electrical'].isna()])
elec=df[df['YearBuilt']>2005]
print(elec['Electrical'].value_counts())
df["Electrical"].fillna("SBrkr",inplace=True)

#vérification des valeurs manquantes
#valeur manquante ordonnée
nulls=pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns=['Null Count']
nulls.index.name='Feature'
#nulls
print(nulls)

#recherche valeur manquante Exterior
df[df['Exterior1st'].isna()]
#approximation valeur manquante
sousEnsemble10=df[df['RoofMatl']=='Tar&Grv']
print(sousEnsemble10.shape)
print(sousEnsemble10.head(23))

#estimation de valeur
df["Exterior1st"].fillna("Wd Sdng",inplace=True)
df["Exterior2nd"].fillna("Wd Sdng",inplace=True)
#pour le type de vente on remplace par le mod
df["SaleType"].fillna("WD",inplace=True)
#pour la qualitée de la cuisine on fait de même
df["KitchenQual"].fillna("TA",inplace=True)
#On gardera les 157 valeurs manquantes en vue de les changer après le preproccesing par une valeur négative

#Stockage du jeu de donnée quasi clean
df.to_csv('predClean.csv', sep=';')
