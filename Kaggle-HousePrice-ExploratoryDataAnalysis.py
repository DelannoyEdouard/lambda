# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 08:37:52 2022

@author: edoua
"""

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

#répartition des valeurs numériques
print(df.describe())
#valeur manquante ordonnée
nulls=pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns=['Null Count']
nulls.index.name='Feature'
#nulls
print(nulls)

""" Prix de vente de logement en Iowa à Ames de 2006 à 2010.Description du jeu de donnée de 80 colonnes avec 2919 valeurs.
Les colonnes sont :

1/MSSubClass: Identifie le type de logement concerné par la vente.
        20	1-STORY 1946 & NEWER ALL STYLES                                                            Occurences:1079	36.96%
        30	1-STORY 1945 & OLDER                                                                       Occurences:139	4.76%
        40	1-STORY W/FINISHED ATTIC ALL AGES                                                          Occurences:6 	0.21%
        45	1-1/2 STORY - UNFINISHED ALL AGES                                                          Occurences:18	0.62%
        50	1-1/2 STORY FINISHED ALL AGES                                                              Occurences:287	9.83%
        60	2-STORY 1946 & NEWER                                                                       Occurences:575	19.70%
        70	2-STORY 1945 & OLDER                                                                       Occurences:128	4.39%
        75	2-1/2 STORY ALL AGES                                                                       Occurences:23	0.79%
        80	SPLIT OR MULTI-LEVEL                                                                       Occurences:118	4.04%
        85	SPLIT FOYER                                                                                Occurences:48	1.64%
        90	DUPLEX - ALL STYLES AND AGES                                                               Occurences:109	3.73%
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER                                      Occurences:182	6.24%
       150	1-1/2 STORY PUD - ALL AGES                                                                 Occurences:1 	0.03%                                                                                  Occurences:63	
       160	2-STORY PUD - 1946 & NEWER                                                                 Occurences:128	4.39%
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER                                                    Occurences:17	0.58%
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES                                                  Occurences:61	2.09%

2/MSZoning: Identifie la classification générale de zonage de la vente.    np.Nan:5
       RL   	Residential Low Density                                                                Occurences:2265	77.70%
       RM   	Residential Medium Density                                                             Occurences:460	15.78%       
       FV   	Floating Village Residential                                                           Occurences:139	4.77%       
       RH   	Residential High Density                                                               Occurences:26	0.89%       
       C (all)	Commercial                                                                             Occurences:25	0.86%

       
3/LotFrontage: Distance entre la rue et le logement en pieds carrés. [21;313] (Continue) np.Nan:486 [25%:59|50%:68|75%:80]

4/LotArea: Surface du terrain en pieds carrés. [1300;215245] (Continue)                            [25%:7478|50%:9453|75%:11570]

5/Street: Type d'accès routier à la propriété.
       Pave 	Pavé                                                                                   Occurences:2907	99.59%       
       Grvl 	Gravier                                                                                Occurences:12	0.41%

       
6/Alley:Type de ruelle d'accès à la propriété.
       NA   	pas d'accés à la ruelle                                                                Occurences:2721	93.22%       
       Grvl 	Gravier                                                                                Occurences:120	4.11%
       Pave 	Pavé                                                                                   Occurences:78	2.67%
       
7/LotShape: Forme de la Propriété.
       Reg	Régulier                                                                                   Occurences:1859	63.69%
       IR1	Légèrement irrégulier                                                                      Occurences:968	33.16%
       IR2	Modérément Irrégulier                                                                      Occurences:76	2.60%
       IR3	Irrégulier                                                                                 Occurences:16	0.55%
    
8/LandContour: planéité du terrain.
       Lvl	Proche du plat/niveau                                                                      Occurences:2622	89.83%
       HLS	Hillside - Pente importante d'un côté à l'autre                                            Occurences:120	4.11%
       Bnk	Montée rapide et significative du niveau de la rue au bâtiment                             Occurences:117	4.01%
       Low	Dépression                                                                                 Occurences:60	2.06%
       
9/Utilities: Type de service disponible.      np.Nan:2
       AllPub	Tous les services publics (Elec, Gaz, Water et Septique(fosse))                        Occurences:2916	99.91%
       NoSeWa	Electricité et Gaz Uniquement                                                          Occurences:1 	0.03%

10/LotConfig: Configuration du lot
       Inside	À l'intérieur du terrain                                                               Occurences:2133	73.07%
       Corner	Terrain de coin                                                                        Occurences:511	17.51%
       CulDSac	Cul de sac                                                                             Occurences:176	6.03%
       FR2  	Façade sur 2 côtés de la propriété                                                     Occurences:85	2.91%
       FR3  	Façade sur 3 côtés de la propriété                                                     Occurences:14	0.48%
       
11/LandSlope: Pente du terrain.
       Gtl	Pente douce                                                                                Occurences:2778	95.17%
       Mod	Pente Modérée                                                                              Occurences:125	4.28%
       Sev	Pente sévère                                                                               Occurences:16	0.55%
       
12/Neighborhood: Emplacements physiques dans les limites de la ville d'Ames
       NAmes	North Ames                                                                             Occurences:443	15.18%
       CollgCr	College Creek                                                                          Occurences:267	9.15%      
       OldTown	Old Town                                                                               Occurences:239	8.19% 
       Edwards	Edwards                                                                                Occurences:194	6.65% 
       Somerst	Somerset                                                                               Occurences:182	6.24%
       NridgHt	Northridge Heights                                                                     Occurences:166	5.69%
       Gilbert	Gilbert                                                                                Occurences:165	5.65%
       Sawyer	Sawyer                                                                                 Occurences:151	5.17%
       NWAmes	Northwest Ames                                                                         Occurences:131	4.49%
       SawyerW	Sawyer West                                                                            Occurences:125	4.28%
       Mitchel	Mitchell                                                                               Occurences:114	3.90%
       BrkSide	Brookside                                                                              Occurences:108	3.70%
       Crawfor	Crawford                                                                               Occurences:103	3.53%
       IDOTRR	Iowa DOT and Rail Road                                                                 Occurences:93	3.19%
       Timber	Timberland                                                                             Occurences:72	2.47%
       NoRidge	Northridge                                                                             Occurences:71	2.43%
       StoneBr	Stone Brook                                                                            Occurences:51	1.75%
       SWISU	South & West of Iowa State University                                                  Occurences:48	1.64%
       ClearCr	Clear Creek                                                                            Occurences:44	1.51%
       MeadowV	Meadow Village                                                                         Occurences:37	1.27%
       BrDale	Briardale                                                                              Occurences:30	1.03%
       Blmngtn	Bloomington Heights                                                                    Occurences:28	0.96%
       Veenker	Veenker                                                                                Occurences:24	0.82%
       NPkVill	Northpark Villa                                                                        Occurences:23	0.79%
       Blueste	Bluestem                                                                               Occurences:10	0.34%
       
13/Condition1: Proximité d'un axe de circulation.
       Norm 	Normale                                                                                Occurences:2511	86.02%       
       Feedr	Adjacente à la rue principale                                                          Occurences:164	5.62%       
       Artery	Adjacente à un axe routier                                                             Occurences:92	3.15%
       RRAn 	Adjacent au chemin de fer nord-sud                                                     Occurences:50	1.71%
       PosN 	À proximité d'un élément positif hors site - parc, ceinture de verdure, etc.           Occurences:39	1.34%
       RRAe 	adjacent au chemin de fer est-ouest                                                    Occurences:28	0.96%
       PosA 	Adjacent à un élément positif hors site                                                Occurences:20	0.69%
       RRNn 	À moins de 200 pieds du chemin de fer nord-sud                                         Occurences:9 	0.31%
       RRNe 	À moins de 200 pieds du chemin de fer est-ouest                                        Occurences:6 	0.21%
        
14/Condition2: Proximité d'un second axe de circulation.
       Norm 	Normale                                                                                Occurences:2889	98.97%
       Feedr	Adjacente à la rue principale                                                          Occurences:13	0.45%
       Artery	Adjacente à un axe routier                                                             Occurences:5 	0.17%
       PosA 	Adjacent à un élément positif hors site                                                Occurences:4 	0.14%
       PosN 	À proximité d'un élément positif hors site - parc, ceinture de verdure, etc.           Occurences:4 	0.14%
       RRNn 	À moins de 200 pieds du chemin de fer nord-sud                                         Occurences:2 	0.07%
       RRAn 	Adjacent au chemin de fer nord-sud                                                     Occurences:1 	0.03%
       RRAe 	Adjacent au chemin de fer est-ouest                                                    Occurences:1 	0.03%
       
15/BldgType: Type de logement.
       1Fam 	Famille seul détaché                                                                   Occurences:2425	83.06%
       TwnhsE	Unité de bout de maison de ville                                                       Occurences:227	7.78%
       Duplx	Duplex                                                                                 Occurences:109	3.73%
       Twnhs	Maison de ville à l'intérieur de l'unité                                               Occurences:96	3.29%
       2FmCon	Conversion bifamiliale; construit à l'origine comme maison unifamiliale                Occurences:62	2.12%
       
16/HouseStyle: Style d'habitation.
       1Story	1 niveau                                                                               Occurences:1471	50.39%
       2Story	2 niveau                                                                               Occurences:872	29.87%
       1.5Fin	1 niveau et demie: 2ème niveau fini                                                    Occurences:314	10.76%
       SLvl 	Demi niveau                                                                            Occurences:128	4.39%
       SFoyer	Foyer divisé                                                                           Occurences:83	2.84%
       2.5Unf	2 niveau et demie: 3ème niveau pas fini                                                Occurences:24 	0.85%
       1.5Unf	1 niveau et demie: 2ème niveau pas fini                                                Occurences:19	0.65%
       2.5Fin	2 niveau et demie: 3ème niveau fini                                                    Occurences:8 	0.27%
       
17/OverallQual:Évalue le matériau global et la finition de la maison.
       10	Très excellent                                                                             Occurences:31	1.06%
       9	Excellent                                                                                  Occurences:107	3.67%
       8	Très bien                                                                                  Occurences:342	11.72%
       7	Bien                                                                                       Occurences:600	20.56%
       6	Au dessus de la moyenne                                                                    Occurences:731	25.04%
       5	La moyenne                                                                                 Occurences:825	28.26%
       4	En dessous de la moyenne                                                                   Occurences:226	7.74%
       3	Equitable                                                                                  Occurences:40	1.37%
       2	Mauvais                                                                                    Occurences:13	0.45%
       1	Très mauvais                                                                               Occurences:4 	0.14%
       
18/OverAllCond: Evaluation de l'état général du bien.
       9	Excellent                                                                                  Occurences:41	1.40%
       8	Très bien                                                                                  Occurences:144	4.93%
       7	Bien                                                                                       Occurences:390	13.36%
       6	Au dessus de la moyenne                                                                    Occurences:531	18.19%
       5	La moyenne                                                                                 Occurences:1645	56.35%
       4	En dessous de la moyenne                                                                   Occurences:101	3.46%
       3	Equitable                                                                                  Occurences:50	1.71%
       2	Mauvais                                                                                    Occurences:10	0.34%
       1	Très mauvais                                                                               Occurences:7 	0.24%
       
19/YearBuilt: Année de construction. [1872;2010] (Continue)                                        [25%:1954|50%:1973|75%:2001]

20/YearRemodAdd: Année de rénovation. [1950;2010] (Continue)                                       [25%:1965|50%:1993|75%:2004]

21/RoofStyle: Type de toit.
       Gable	Gable                                                                                  Occurences:2310	79.14%
       Hip  	Hip(bas)                                                                               Occurences:551	18.88%
       Gambrel	Gambrel(grange)                                                                        Occurences:22	0.75%
       Flat 	Plat                                                                                   Occurences:20	0.69%
       Mansard	Mansardé                                                                               Occurences:11	0.38%
       Shed 	Hangar                                                                                 Occurences:5 	0.17%
       
22/RoofMatl: Matériel du toit.
       CompShg	Bardeau standard (composite)                                                           Occurences:2876	98.53%
       Tar&Grv	Gravier et goudron                                                                     Occurences:23	0.79%
       WdShake	Bardeau de bois                                                                        Occurences:9 	0.31%
       WdShngl	Bardeaux de bois                                                                       Occurences:7 	0.24%
       ClyTile	Argile ou Tuile                                                                        Occurences:1 	0.03%
       Membran	Membrane                                                                               Occurences:1 	0.03%
       Metal	Metal                                                                                  Occurences:1 	0.03%
       Roll 	Roulé                                                                                  Occurences:1 	0.03%
       
23/Exterior1st: Revêtement extérieur du logement.         np.Nan:1
       VinylSd	Bardage en vinyle                                                                      Occurences:1025	35.13%
       MetalSd	Bardage métallique                                                                     Occurences:450	15.42%
       HdBoard	Panneau dur                                                                            Occurences:442	15.15%
       Wd Sdng	Bardage en bois                                                                        Occurences:411	14.09%
       Plywood	Contre-plaqué                                                                          Occurences:221	7.57%
       CemntBd	Plaque de ciment                                                                       Occurences:126	4.32%
       BrkFace	Visage de brique                                                                       Occurences:87	2.98%
       WdShing	Bardeaux de bois                                                                       Occurences:56	1.92%
       AsbShng	Bardeaux d'amiante                                                                     Occurences:44	1.51%
       Stucco	Stuc                                                                                   Occurences:43	1.47%
       BrkComm	Brique commune                                                                         Occurences:6 	0.21%
       Stone	Pierre                                                                                 Occurences:2 	0.14%
       AsphShn	Bardeaux d'asphalte                                                                    Occurences:2 	0.07%
       CBlock	Parpaing                                                                               Occurences:2 	0.07%
       ImStucc	Imitation Stuc                                                                         Occurences:1 	0.03%
       
24/Exterior2nd: Revêtement extérieur du logement (si plus d'un matériau). np.Nan:1
       VinylSd	Bardage en vinyle                                                                      Occurences:1014	34.75%
       MetalSd	Bardage métallique                                                                     Occurences:447	15.32%
       HdBoard	Panneau dur                                                                            Occurences:406	13.91%
       Wd Sdng	Bardage en bois                                                                        Occurences:391	13.40%
       Plywood	Contre-plaqué                                                                          Occurences:270	9.25%
       CemntBd	Plaque de ciment                                                                       Occurences:126	4.32%
       Wd Shng	Bardeaux de bois                                                                       Occurences:81	2.78%
       Stucco	Stuc                                                                                   Occurences:47	1.61%
       BrkFace	Visage de brique                                                                       Occurences:47	1.61%
       AsbShng	Bardeaux d'amiante                                                                     Occurences:38	1.30%
       Brk Cmm	Brique commune                                                                         Occurences:22	0.75%
       ImStucc	Imitation Stuc                                                                         Occurences:15	0.51%
       Stone	Pierre                                                                                 Occurences:6 	0.21%
       AsphShn	Bardeaux d'asphalte                                                                    Occurences:4 	0.14%
       CBlock	Parpaing                                                                               Occurences:2 	0.07%
       Other	Autre                                                                                  Occurences:1 	0.03%
       
25/MasVnrType: Type de plaquage (maçonnerie). np.Nan:24 	1.20%
       None 	None                                                                                   Occurences:1742	59.68%
       BrkFace	Visage de brique                                                                       Occurences:879	30.11%
       Stone	Pierre                                                                                 Occurences:249	8.53%
       BrkCmn	Brique commune                                                                         Occurences:25	0.86%
       
26/MasVnrArea: Surface de plaquage en pieds carrés. [0;1600] (Continue)  np.Nan:23                   [25%:0|0:50%|75%:164]

27/ExterQual: Evaluation de la qualité des matériaux extérieur. 
       Ex	Excellent                                                                                  Occurences:107	3.67%
       Gd	Bien                                                                                       Occurences:979	33.54%
       TA	La moyenne                                                                                 Occurences:1798	61.60%
       Fa	Equitable                                                                                  Occurences:35	1.20%
       
28/ExterCond: Etat des matériaux extèrieur.
       Ex	Excellent                                                                                  Occurences:12	0.41%
       Gd	Bien                                                                                       Occurences:299	10.24%
       TA	La moyenne                                                                                 Occurences:2538	86.95%
       Fa	Equitable                                                                                  Occurences:67	2.30%
       Po	Mauvais                                                                                    Occurences:3 	0.10%
       
29/Foundation: Type de fondation.
       PConc	Beton coulé                                                                            Occurences:1308	44.81%
       CBlock	Parpaing                                                                               Occurences:1235	42.31%
       BrkTil	Brique et tuile                                                                        Occurences:311	10.65%
       Slab 	Dalle                                                                                  Occurences:49	1.68%
       Wood 	Bois                                                                                   Occurences:11	0.38%
       Stone	Pierre                                                                                 Occurences:5 	0.17%
       
30/BsmtQual: Evalue la hauteur du sous-sol.       
       Ex	Excellent (100+ inches)                                                                    Occurences:258	8.84%
       Gd	Bien (90-99 inches)                                                                        Occurences:1209	41.42%
       TA	La moyenne (80-89 inches)                                                                  Occurences:1283	43.95%
       Fa	Equitable (70-79 inches)                                                                   Occurences:88	3.01%
       NA	Pas de sous-sol                                                                            Occurences:81	2.77%
       
31/BsmtCond: Etat du sous-sol.                     np.Nan:1
       Gd	Bien                                                                                       Occurences:122	4.18%
       TA	La moyenne(légère humidité autorisée)                                                      Occurences:2606	89.28%
       Fa	Equitable( humidité ou fissures ou tassements)                                             Occurences:104	3.56%
       Po	Mauvais(Fissuration, sédimentation ou humidité sévères)                                    Occurences:5 	0.17%
       NA	Pas de sous-sol                                                                            Occurences:81	2.77%
       
32/BsmtExposure: Mur du sous-sol ou du rez-de-jardin. np.Nan:1
       Gd	Bonne exposition                                                                           Occurences:276	9.46%
       Av	Exposition moyenne (les niveaux divisés ou les foyers obtiennent généralement un score moyen ou supérieur
                                                                                                       Occurences:418	14.32%
       Mn	Exposition minimal                                                                         Occurences:239	8.19%
       No	Pas exposition                                                                             Occurences:1904	65.23%
       NA	Pas de sous-sol                                                                            Occurences:81	2.77%
       
33/BsmtFinType1: Évaluation de la superficie finie du sous-sol. 
       GLQ	Bonne pièce d'habitation                                                                   Occurences:849	29.09%
       ALQ	Moyenne pièce d'habitation                                                                 Occurences:429	14.70%
       BLQ	Pièce en dessous de la moyenne                                                             Occurences:269	9.22%
       Rec	Salle de jeux moyenne                                                                      Occurences:288	9.87%
       LwQ	Qualité basse                                                                              Occurences:154	5.28%
       Unf	Pas fini                                                                                   Occurences:851	29.15%
       NA	Pas de sous-sol                                                                            Occurences:81	2.77%
       
34/BsmtFinSF1: Évaluation de la superficie finie du sous-sol en pieds carrés. [0;5644] (Continue)  [25%:0|50%:369|75%:733]

35/BsmtFinType2: Évaluation de la surface finie du sous-sol (si plusieurs types).
       GLQ	Bonne pièce d'habitation                                                                   Occurences:34	1.16%
       ALQ	Moyenne pièce d'habitation                                                                 Occurences:52	1.78%
       BLQ	Pièce en dessous de la moyenne                                                             Occurences:68	2.33%
       Rec	Salle de jeux moyenne                                                                      Occurences:105	3.60%
       LwQ	Qualité basse                                                                              Occurences:87	2.98%
       Unf	Pas fini                                                                                   Occurences:2493	85.41%
       NA	Pas de sous-sol ou pas de 2nd type                                                         Occurences:81	2.77%
       
36/BsmtFinSF2: Évaluation de la surface finie du sous-sol en pieds carrés (si plusieurs types). [0;1474] (Continue)
                                                                                                    [25%:0|50%:0|75%:0]

37/BsmtUnfSF:Pieds carrés non fini du sous-sol. [0;2336] (Continue)                                 [25%:220|50%:467|75%:806]

38/TotalBsmtSF:Total de pieds carrés de sous-sol. [0;6110] (Continue)                               [25%:793|50%:990|75%:1302]

39/Heating: Type de chauffage.
       GasA 	Fournaise à air chaud pulsé au gaz                                                     Occurences:2874	98.46%
       GasW 	Eau chaude au gaz ou chauffage vapeur                                                  Occurences:27	0.93%
       Grav 	Four à gravité                                                                         Occurences:9 	0.31%
       Wall 	Fournaise murale                                                                       Occurences:6 	0.21%
       OthW 	Chauffage à l'eau chaude ou à la vapeur autre que le gaz                               Occurences:2 	0.07%
       Floor	Fournaise au sol                                                                       Occurences:1 	0.03%
       
40/HeatingQC: Qualité et état du chauffage.
       Ex	Excellent                                                                                  Occurences:1493	51.15%
       Gd	Bien                                                                                       Occurences:474	16.24%
       TA	La moyenne                                                                                 Occurences:857	29.36%
       Fa	Equitable                                                                                  Occurences:92	3.15%
       Po	Mauvais                                                                                    Occurences:3 	0.10%
       
41/CentralAir: Climatisation Centrale. (Binaire)
       N	No                                                                                         Occurences:196	6.71%
       Y	Yes                                                                                        Occurences:2723	93.29%
       
42/Electrical: Système electrique. np.Nan:1
       SBrkr	Disjoncteurs standards et Romex                                                        Occurences:2671	91.54%
       FuseA	Boîte à fusibles de plus de 60 ampères et tout le câblage Romex (moyenne)              Occurences:188	6.44%
       FuseF	Boîte à fusibles 60 ampères et principalement câblage Romex (juste)                    Occurences:50	1.71%
       FuseP	Boîte à fusibles 60 A et principalement le câblage des boutons et des tubes (mauvais)  Occurences:8 	0.27%
       Mix  	Système hybride                                                                        Occurences:1 	0.03%
       
43/1stFlrSF: Surface du 1er étage(Rdc) en pieds carrés. [334;5095] Continue                        [25%:876|50%/1082|75%:1387]

44/2ndFlrSF: Surface du 2nd étage(1er) en pieds carrés. [0;2065] Continue                          [25%:0|50%:0|75%:704]

45/LowQualFinSF: Pieds carrés finis de mauvaise qualité (tous les étages). [0;1064] (Continue)      [25%:0|50%:0|75%:0]

46/GrLiveArea:Surface habitable en pieds carrés (hors sous-sol). [334;5642] (Continue)             [25%:1126|50%:1444|75%:1744]

47/BsmtFullBath: Salle de bain complête en sous-sol. [0;3] (Ordinal)             
                                   [0:1705	58.41%|1:1172	40.15%|2:38 	1.30%|3:2	0.07%]

48/BsmtHalfBath: Demie salle de bain en sous-sol. [0;2] (Ordinal)                                  
                                   [0:2742	93.94%|1:171 	5.86%|2:4	0.14%]

49/FullBath: Salle de bain complête (hors sous-sol). [0;4] (Ordinal)                               
       0:12 	0.41%
       1:1309	44.84%
       2:1530	52.42%
       3:64 	2.19%
       4:4  	0.14%

50/HalfBath: Demie salle de bain (hors sous-sol). [0;2] (Ordinal)                                 
                                   [0:1834	62.83%|1:1060	36.31%|2:125 	0.86%]

51/BedroomAbvGr: Nombre de chambre (hors sous-sol). [0;8] (Ordinal)                                
       0:8  	0.27%
       1:103	3.53%
       2:742	25.42%
       3:1596	54.68%
       4:400	13.70%
       5:48 	1.64%
       6:21  	0.72%
       8:1  	0.03%

52/KitchenAbvGr:Nombre de cuisine (hors sous-sol). [0;3] (Ordinal)                                 
                                   [0:1 	0.07%|1:1392	95.34%|2:65 	4.45%|3:2	0.14%]

53/KitchenQual: Qualité cuisine.
       Ex	Excellent                                                                                  Occurences:205	7.03%
       Gd	Bien                                                                                       Occurences:1151	39.44%
       TA	La moyenne                                                                                 Occurences:1492	51.13%
       Fa	Equitable                                                                                  Occurences:70	2.40%
       
54/TotRmsAbvGrd: Nombre de pièces (hors salle de bain). [2;15] (Ordinal)
       2:1  	0.03%
       3:25 	0.86%
       4:196	6.71%
       5:583	19.97%
       6:844	28.91%
       7:649	22.23%
       8:347	11.89%
       9:143	4.90%
       10:80	2.74%
       11:32	1.10%
       12:16	0.55%
       14:1 	0.03%
       15:1 	0.03%

55/Functional:Fonctionnalité d'accueil (Supposer typique à moins que des déductions ne soient justifiées).
       Typ  	Fonctionnalité typique                                                                 Occurences:2717	93.08%
       Min2 	Déductions mineures 2                                                                  Occurences:70	2.40%
       Min1 	Déductions mineures 1                                                                  Occurences:65	2.23%
       Mod  	Déductions modérées                                                                    Occurences:35	1.20%
       Maj1 	Déductions majeures 1                                                                  Occurences:19	0.65%
       Maj2 	Déductions majeures 2                                                                  Occurences:9 	0.31%
       Sev  	Gravement endommagé                                                                    Occurences:2 	0.07%
       
56/Fireplaces: Nombre de cheminée. [0;4] (Ordinal)
       0:1420	48.65%
       1:1268	43.44%
       2:219	7.50%
       3:11 	0.38%
       4:1  	0.03%

57/FireplaceQu: Qualité cheminée.
       Ex	Excellent                                                                                  Occurences:43	1.47%
       Gd	Bien                                                                                       Occurences:744	25.49%
       TA	La moyenne                                                                                 Occurences:592	20.28%
       Fa	Equitable                                                                                  Occurences:74	2.54%
       Po	Mauvais                                                                                    Occurences:46	1.58%
       NA	Pas de cheminée                                                                            Occurences:1420	48.65%
       
58/GarageType: Localisation du garage.
       Attchd	Attaché à la maison                                                                    Occurences:1723	59.03%%
       Detchd	Détaché de la maison                                                                   Occurences:779	26.69%
       BuiltIn	Intégré (partie garage de la maison - a généralement de la place au-dessus du garage)  Occurences:186	6.37%
       NA   	Pas de garage                                                                          Occurences:157	5.38%
       Basment	Garage au sous-sol                                                                     Occurences:36	1.23%
       2Types	Plusieurs types de garage                                                              Occurences:23	0.79%
       CarPort	Abri de voiture                                                                        Occurences:15	0.51%
       
59/GarageYrBlt: Année de construction du garage. [1900;2010] (Continue)                            [25%:1960|50%:1979|75%:2002]

60/GarageFinish: Finition du garage.
       Unf	Inachevé                                                                                   Occurences:1230	42.14%
       RFn	Rugueux fini                                                                               Occurences:811	27.78%
       Fin	Achevé                                                                                     Occurences:719	24.63%
       NA	Pas de garage                                                                              Occurences:157	5.38%
       
61/GarageCars: Capacité de stockage de voiture. [0;5] (Ordinal)                                    
       0:157	5.38%
       1:776	26.59%
       2:1594	54.63%
       3:374	12.82%
       4:16 	0.55%
       5:1  	0.03%

62/GarageArea: Taille du garage en pieds carrés. [0;1488] (Continue)                               [25%:320|50%:480|75%:576]

63/GarageQual: Qualité du garage.
       Ex	Excellent                                                                                  Occurences:3 	0.10%
       Gd	Bien                                                                                       Occurences:24	0.82%
       TA	La moyenne                                                                                 Occurences:2604	89.21%
       Fa	Equitable                                                                                  Occurences:124	4.25%
       Po	Mauvais                                                                                    Occurences:5 	0.17%
       NA	Pas de garage                                                                              Occurences:157	5.38%
       
64/GarageCond: Condition du garage.
       Ex	Excellent                                                                                  Occurences:3 	0.10%
       Gd	Bien                                                                                       Occurences:15	0.51%
       TA	La moyenne                                                                                 Occurences:2654	90.92%
       Fa	Equitable                                                                                  Occurences:74	2.54%
       Po	Mauvais                                                                                    Occurences:14	0.48%
       NA	Pas de garage                                                                              Occurences:157	5.38%
       
65/PavedDrive:Allée pavé
       Y	Pavé                                                                                       Occurences:2641	90.48%
       P	Partiellement Pavé                                                                         Occurences:62	2.12%
       N	Terre/gravier                                                                              Occurences:216	7.40%
       
66/WoodeckSF: Superficie de la terrasse en bois en pieds carrés. [0;1424] (Continue)                 
                                   [25%:0|50%:0|75%:168]

67/OpenPorchSF: Superficie porche ouvert en pieds carrés. [0;742] (Continue)                        
                                   [25%:0|50%:26|75%:70]

68/EnclosedPorch: Superficie porche fermé en pieds carrés. [0;1012] (Continue)                       
                                   [25%:0|50%:0|75%:0]

69/3SsnPorch: Superficie porche 3 saisons en pieds carrés. [0;508] (Continue)                       
                                   [25%:0|50%:0|75%:0]

70/ScreenPorch: Superficie porche moustiquaire en pieds carrés. [0;576] (Continue)                  
                                   [25%:0|50%:0|75%:0]

71/PoolArea: Superficie piscine en pieds carrés. [0;800] (Ordinal)
       0:2906	99.52%
       144:1	0.03%
       228:1	0.03%
       368:1	0.03%
       444:1	0.03%
       480:1	0.03%
       512:1	0.03%
       519:1	0.03%
       555:1	0.03%
       561:1	0.03%
       576:1	0.03%
       648:1	0.03%
       738:1	0.03%
       800:1	0.03%

72/PoolQC: Qualité de la piscine. np.Nan:3
       Ex	Excellent                                                                                  Occurences:4 	0.14%
       Gd	Bien                                                                                       Occurences:4 	0.14%
       Fa	Equitable                                                                                  Occurences:2 	0.07%
       NA	Pas de piscine                                                                             Occurences:2906	99.55%
       
73/Fence: Qualité de la cloture
       MnPrv	Minimum Privacy                                                                        Occurences:329	11.27%
       GdPrv	Good Privacy                                                                           Occurences:118	4.04%
       GdWo 	Good Wood                                                                              Occurences:112	3.84%
       MnWw 	Minimum Wood/Wire                                                                      Occurences:12	0.41%
       NA   	Pas de cloture                                                                         Occurences:2348	80.44%
       
74/MiscFeature: Fonctionnalité diverse non couverte dans les autres catégories.
       Shed 	Cabanon (over 100 SF)                                                                  Occurences:95	3.25%
       Gar2 	2e garage (si non décrit dans la section garage)                                       Occurences:5 	0.17%
       Othr 	Autre                                                                                  Occurences:4 	0.14%
       TenC 	Court de tennis                                                                        Occurences:1 	0.03%
       NA   	Aucun                                                                                  Occurences:2814	96.40%
       
75/MiscVal: Valeurs en dollar des fonctionnalités diverses non couverte dans les autres catégories. [0;17000] (Continue)
                                                                                                   [25%:0|50%:0|75%:0] 

76/MoSold: Mois vendu. [1;12] (Ordinal)
       1:122	4.18%
       2:133	4.56%
       3:232	7.95%
       4:279	9.56%
       5:394	13.50%
       6:503	17.23%
       7:446	15.28%
       8:233	7.98%
       9:158	5.41%
       10:173	5.93%
       11:142	4.86%
       12:104	3.56%

77/YrSold: Année vendu. [2006;2010] (Ordinal)                                                      
       2006:619 	21.21%
       2007:692 	23.71%
       2008:622 	21.31%
       2009:647 	22.17%
       2010:339 	11.61%

78/SaleType: Type de vente.   np.Nan:1
       WD   	Acte de garantie - Conventionnel                                                       Occurences:2525	86.50%
       New  	Maison juste construite et vendue                                                      Occurences:239	8.19%
       COD  	Acte d'officier de justice/succession                                                  Occurences:87	2.98%
       ConLD	Contrat Low Down                                                                       Occurences:26	0.89%
       CWD  	Acte de garantie - Espèces                                                             Occurences:12	0.41%
       ConLI	Contrat à faible intérêt                                                               Occurences:9 	0.31%
       ConLw	Contrat Low Down Acompte et faible taux d'intérêt                                      Occurences:8 	0.27%
       Oth  	Autre                                                                                  Occurences:7 	0.24%
       Con  	Contrat 15% Acompte conditions régulières                                              Occurences:5 	0.17%
       
79/SaleCondition: Condition de vente.
       Normal	Vente normale                                                                          Occurences:2402	82.29%
       Partial	La maison n'était pas terminée lors de la dernière évaluation (associée aux nouvelles maisons) 
                                                                                                       Occurences:245	8.39%
       Abnorml	Vente anormale - commerce, forclusion, vente à découvert                               Occurences:190	6.51%
       Family	Vente entre membres de la famille                                                      Occurences:46	1.58%
       Alloca	Attribution - deux propriétés liées avec des actes séparés, généralement un condo avec une unité de garage 
                                                                                                       Occurences:24	0.82%
       AdjLand	Achat d'un terrain attenant                                                            Occurences:12	0.41%

       
80/SalePrice: Prix de vente. [34900;755000] (Continue)                                        [25%:129975|50%:163000|75%:214000]
"""

#Visualisation des relations entres colonnes numériques
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True, cmap='Greens')
#Classement des colonnes numériques corrélées avec la target
print(df.corr()['SalePrice'].sort_values(ascending=False))

print(#Comparont les relations entre la surface et les autres colonnes numériques
df.corr()['GrLivArea'].sort_values(ascending=False))

"""-On constate une corrélation élévée entre la surface habitable(GrLivArea) et le prix : 0.59 et l'évaluation globale des 
    matériaux plus finitions(OverallQual) et le prix : 0.55
   -On remarque aussi une forte relation entre les colonnes: 
      .GarageArea et GarageCars; 0.89  (qui délivre la même information)
      .GarageYrBlt et YearBuilt; 0.83
      .TotRmsAbvGrd et GrLivArea; 0.81  (corrélation évidente entre le nombre de piece et la surface habitable)
      .1stFlrSFC et TotalBsmtSF; 0.80  (corrélation rassurante car la base est du même ordre que le rez-de-chausé)"""
      
#Visualisation des colonnes les plus corrélées avec le prix
plt.figure(figsize=[10,10])
plt.subplot(2,1,1)
sns.scatterplot(x='GrLivArea',y="SalePrice",data=df)
plt.title('prix/surfaceHabitable')
plt.subplot(2,1,2)
sns.boxplot(x='OverallQual',y="SalePrice",data=df)
plt.title('prix/qualiteMateriaux')

#Comparaison du prix de vente en fonction de la surface habitable et du type de batîment
sns.lmplot(x='GrLivArea',y="SalePrice",hue='BldgType',data=df,height=6,aspect=1.5)
plt.title('prix/surfaceHabitable en fonction du type de logement')

#Comparons l'évolution des prix en fonction de l'année sachant que la crise des subprimes à eu un impact sur l'immobilier
#entre 2006 et 2010

plt.figure(figsize=(12,8))
sns.lineplot(x='GrLivArea',y='SalePrice',hue='YrSold',data=df,palette="tab10",linewidth=2.5)
plt.title('prix/surfaceHabitable en fonction de l année')

g=sns.lmplot(x="GrLivArea",y="SalePrice",hue="YrSold",col="YrSold",col_wrap=3,
               data=df,height=6,aspect=1)

#Une tendance tend à penser que le prix augmente plus la construction est récente avec une certaine homoscédascité.
plot = plot = df.groupby("YearBuilt")["SalePrice"].mean()
plt.scatter(np.unique(df["YearBuilt"]), plot)
plt.xlabel("Année")
plt.ylabel("prix de vente moyen")
plt.show()

#Création de 2 sous ensembles en stockant les 25% des valeurs ou la target est élévée ou basse en vue de les comparer
df_high=df[df['SalePrice']>214000]
df_min=df[df['SalePrice']<=129975]
#Comparaison des valeurs ordinales
ordinal=df[['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
            'Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond','RoofStyle','RoofMatl','Exterior1st',
            'Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
            'BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType',
            'GarageFinish','GarageCars','GarageQual','GarageCond','PavedDrive','PoolArea','PoolQC','Fence','MiscFeature',
            'MoSold','YrSold','SaleType','SaleCondition']]
for col in ordinal:
    plt.figure(figsize=(20,15))
    plt.subplot(1,3,1)
    df[col].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('df')
    plt.subplot(1,3,2)
    df_high[col].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('df SalePrice>75')
    plt.subplot(1,3,3)
    df_min[col].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('df SalePrice<25')
    
"""On distinge une forte disparitée entre le style d'habitation, le quartier, la zone et la qualitée des préstations en 
   fonction du prix."""
   
#Comparaison des valeurs continues
continu=['LotFrontage','LotArea','YearBuilt','YearRemodAdd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF',
          '2ndFlrSF','LowQualFinSF','GrLivArea','GarageYrBlt','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
          '3SsnPorch','ScreenPorch','MiscVal']

for col in continu:
    plt.figure(figsize=(12,8))
    sns.distplot(df_high[col],label='df>75')
    sns.distplot(df_min[col],label='df<25')
    plt.legend()
    
#Elaboration d'une carte pour visualiser les quartiers de la ville Ames en Iowa
import folium
import json

m=folium.Map(location=[42.028,-93.620369], zoom_start=13)

marker = folium.Marker(location=[41.991465821432186,-93.60494613691118],
    icon=folium.DivIcon(html=f"""
      <div style="color:#f00;background:#fff;width:60px;text-align:center;">MeadowV</div>
    """))
marker.add_to(m)
marker1 = folium.Marker(location=[42.0017302146107,-93.60893726359792],
    icon=folium.DivIcon(html=f"""
      <div style="color:#ABBAEA;background:#fff;width:60px;text-align:center;">Mitchel</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[41.99434375968837,-93.64877700892976],
    icon=folium.DivIcon(html=f"""
      <div style="color:#df6d14;background:#fff;width:60px;text-align:center;">Timber</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.00046733400666,-93.64740371791413],
    icon=folium.DivIcon(html=f"""
      <div style="color:#f56642;background:#fff;width:60px;text-align:center;">GnrHill</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.00788679355376,-93.64422798178566],
    icon=folium.DivIcon(html=f"""
      <div style="color:#f5d742;background:#fff;width:60px;text-align:center;">Blueste</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.01381775255734,-93.64289760611427],
    icon=folium.DivIcon(html=f"""
      <div style="color:#bcf542;background:#fff;width:60px;text-align:center;">Crawfor</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.0197800409572,-93.6518669130601],
    icon=folium.DivIcon(html=f"""
      <div style="color:#42f54b;background:#fff;width:60px;text-align:center;">SWISU</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.02051333194248,-93.66886138937843],
    icon=folium.DivIcon(html=f"""
      <div style="color:#42f5ec;background:#fff;width:60px;text-align:center;">Edwards</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.01758011726449,-93.68774414084331],
    icon=folium.DivIcon(html=f"""
      <div style="color:#4251f5;background:#fff;width:60px;text-align:center;">CollgCr</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.021278496128765,-93.6280918123521],
    icon=folium.DivIcon(html=f"""
      <div style="color:#a142f5;background:#fff;width:60px;text-align:center;">IDOTRR</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.02941144960496,-93.6198806767061],
    icon=folium.DivIcon(html=f"""
      <div style="color:#f542dd;background:#fff;width:60px;text-align:center;">OldTown</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.03079279601155,-93.6290645603731],
    icon=folium.DivIcon(html=f"""
      <div style="color:#42f5ce;background:#fff;width:60px;text-align:center;">BrkSide</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.02583740853231,-93.6755132677354],
    icon=folium.DivIcon(html=f"""
      <div style="color:#429cf5;background:#fff;width:60px;text-align:center;">ClearCr</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.033045388837856,-93.67584228559282],
    icon=folium.DivIcon(html=f"""
      <div style="color:#752d13;background:#fff;width:60px;text-align:center;">Sawyer</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.03572289533336,-93.6883735661104],
    icon=folium.DivIcon(html=f"""
      <div style="color:#aa8f73;background:#fff;width:60px;text-align:center;">SawyerW</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.03839663690869,-93.6526823046006],
    icon=folium.DivIcon(html=f"""
      <div style="color:#8ea592;background:#fff;width:60px;text-align:center;">Veenker</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.04340051107406,-93.64942073843852],
    icon=folium.DivIcon(html=f"""
      <div style="color:#d5848b;background:#fff;width:60px;text-align:center;">Green</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.04130764316816,-93.62680435202492],
    icon=folium.DivIcon(html=f"""
      <div style="color:#f8bba0;background:#fff;width:60px;text-align:center;">NAmes</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.047840986236714,-93.64856243155373],
    icon=folium.DivIcon(html=f"""
      <div style="color:#76769e;background:#fff;width:60px;text-align:center;">Somerst</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.050539114326334,-93.63826274893657],
    icon=folium.DivIcon(html=f"""
      <div style="color:#690000;background:#fff;width:60px;text-align:center;">NWAmes</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.04974243498062,-93.62886428854839],
    icon=folium.DivIcon(html=f"""
      <div style="color:#8ea592;background:#fff;width:60px;text-align:center;">NPkVill</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.052831669369134,-93.62790584575124],
    icon=folium.DivIcon(html=f"""
      <div style="color:#f4c2c2;background:#fff;width:60px;text-align:center;">BrDale</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.052799803537326,-93.6568951607842],
    icon=folium.DivIcon(html=f"""
      <div style="color:#a742f5;background:#fff;width:60px;text-align:center;">NoRidge</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.059932040193296,-93.65425586711355],
    icon=folium.DivIcon(html=f"""
      <div style="color:#f56942;background:#fff;width:60px;text-align:center;">NridgHT</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.057324591609564,-93.64698171626516],
    icon=folium.DivIcon(html=f"""
      <div style="color:#c8f542;background:#fff;width:60px;text-align:center;">Gilbert</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.05896554424454,-93.63498687755056],
    icon=folium.DivIcon(html=f"""
      <div style="color:#42f5a7;background:#fff;width:60px;text-align:center;">StoneBr</div>
    """))
marker1.add_to(m)
marker1 = folium.Marker(location=[42.06240662760382,-93.64114522944875],
    icon=folium.DivIcon(html=f"""
      <div style="color:#211812;background:#fff;width:60px;text-align:center;">Blmngtn</div>
    """))
marker1.add_to(m)
m

#comparont la répartition du prix en fonction du quartier
df['Feet/Price']=df['SalePrice']/df['GrLivArea']
plt.figure(figsize=(12,8))
fig=sns.boxplot(x='Neighborhood',y='Feet/Price',data=df).set_title('prices per square foot')
plt.xticks(rotation=45)

#création de sous ensemble en fonction des années de vente
df06=df[df['YrSold']==2006]
df07=df[df['YrSold']==2007]
df08=df[df['YrSold']==2008]
df09=df[df['YrSold']==2009]
df10=df[df['YrSold']==2010]
#Visualisation de l'évolultion des prix par quartiers en fonction de l'année
order=df['Neighborhood'].unique()
plt.figure(figsize=[12,22])
plt.subplot(5,1,1)
fig=sns.boxplot(x='Neighborhood',y='Feet/Price',data=df06,order=order).set_title('prices per square foot 2006')
plt.xticks(rotation=45)
plt.subplot(5,1,2)
fig=sns.boxplot(x='Neighborhood',y='Feet/Price',data=df07,order=order).set_title('prices per square foot 2007')
plt.xticks(rotation=45)
plt.subplot(5,1,3)
fig=sns.boxplot(x='Neighborhood',y='Feet/Price',data=df08,order=order).set_title('prices per square foot 2008')
plt.xticks(rotation=45)
plt.subplot(5,1,4)
fig=sns.boxplot(x='Neighborhood',y='Feet/Price',data=df09,order=order).set_title('prices per square foot 2009')
plt.xticks(rotation=45)
plt.subplot(5,1,5)
fig=sns.boxplot(x='Neighborhood',y='Feet/Price',data=df10,order=order).set_title('prices per square foot 2010')
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.1,    
                    top=0.9,  
                    wspace=0.9,  
                    hspace=0.9) 
plt.show()


      
      