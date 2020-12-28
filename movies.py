# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 04:41:54 2020

@author: shara
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
movies = pd.read_csv("F:\Warun\\DS Assignments\\DS Assignments\\Association Rules\\my_movies.csv")
movies.head()
movies.values
movies1 = movies.iloc[:, [5,6,7,8,9,10,11,12,13,14]]
movies1.head
frequent_movies = apriori(movies1, min_support = 0.005, max_len = 3, use_colnames = True)
print(frequent_movies.sort_values("support", ascending = False, inplace = True))
import matplotlib.pyplot as plt
rules = association_rules(frequent_movies, metric = "lift", min_threshold= 1)
rules_r = rules.sort_values('lift', ascending = False)


frequent_movies1 = apriori(movies1, min_support = 0.01, max_len = 4, use_colnames = True)
rules1 = association_rules(frequent_movies1, metric = "lift", min_threshold= 1)

frequent_movies2 = apriori(movies1, min_support = 0.1, max_len = 3, use_colnames = True)
rules2 = association_rules(frequent_movies2, metric = "lift", min_threshold= 1)

frequent_movies3 = apriori(movies1, min_support = 0.1, max_len = 3, use_colnames = True)
rules3 = association_rules(frequent_movies3, metric = "lift", min_threshold= 1.5)

frequent_movies4 = apriori(movies1, min_support = 0.5, max_len = 3, use_colnames = True)
rules4 = association_rules(frequent_movies4, metric = "lift", min_threshold= 1)

frequent_movies5 = apriori(movies1, min_support = 0.007, max_len = 3, use_colnames = True)
rules5 = association_rules(frequent_movies5, metric = "confidence", min_threshold= 0.8)



plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(rules["support"], rules["lift"], alpha=0.5)
plt.xlabel("support")
plt.ylabel("lift")
plt.title("Support vs Lift")
plt.show()


plt.scatter(rules["confidence"], rules["lift"], alpha=0.5)
plt.xlabel("confidence")
plt.ylabel("lift")
plt.title("confidence vs Lift")
plt.show()


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], fit_fn(rules['lift']))