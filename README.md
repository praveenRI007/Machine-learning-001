# Machine-learning
Life expectancy analysis using UCI repository :
this UCI repopsitory contains dataset with 20 features accounting to a total of almost 3000.
the features are:
Country,Year,Status,Life expectancy,Adult Mortality,infant deaths,Alcohol,percentage expenditure,Hepatitis B,Measles , BMI ,under-five deaths ,Polio,Total expenditure,Diphtheria , HIV/AIDS,GDP,Population, thinness  1-19 years, thinness 5-9 years,Income composition of resources,Schooling
In this ML problem our goal is to predict life expectancy from given parameters
ML algorith used : RandomForestRegressor
train/test ratio: 0.8 : 0.2
In our model i remove the 'country' and 'year' as features as i needed to focus upon important aspects disregarding of year/country for future prediction :
from training the model we get the top 5 important feature/aspects which has impact on life expectancy:
(feature's impact level in ascending order)
1.Schooling
2.Adult mortality
3.Income composition of resources
4.BMI
5.HIV/AIDS

Upon training this model we achieve r_2 score of : 0.96 or 96% (accuracy) 
