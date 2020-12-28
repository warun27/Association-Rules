# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:17:01 2020

@author: shara
"""

import pandas as pd
import numpy as np
pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
books = pd.read_csv("F:\\Warun\\DS Assignments\\DS Assignments\\Association Rules\\book.csv")
books.head()
frequent_books_s = apriori(books, min_support = 0.005, max_len = 3, use_colnames = True)
frequent_books_s.sort_values("support", ascending = False, inplace = True)
import matplotlib.pyplot as plt
rules = association_rules(frequent_books_s, metric = "lift", min_threshold= 1)
frequent_books_2 = apriori(books, min_support = 0.005, max_len = 5, use_colnames = True)
rules2 = association_rules(frequent_books_2, metric = "lift", min_threshold= 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(20)
rules_r = rules.sort_values('lift', ascending = False)

    
frequent_books_3 = apriori(books, min_support = 0.005, max_len = 4, use_colnames = True)
rules3 = association_rules(frequent_books_2, metric = "lift", min_threshold= 1)

frequent_books_4 = apriori(books, min_support = 0.01, max_len = 3, use_colnames = True)
rules4 = association_rules(frequent_books_4, metric = "lift", min_threshold= 1.5)

frequent_books_5 = apriori(books, min_support = 0.05, max_len = 3, use_colnames = True)
rules5 = association_rules(frequent_books_5, metric = "confidence", min_threshold= 0.8)

frequent_books_6 = apriori(books, min_support = 0.1, max_len = 3, use_colnames = True)
rules6 = association_rules(frequent_books_6, metric = "confidence", min_threshold= 0.8)

support=rules['support']
confidence=rules['confidence']

plt.bar((range(1,11)), frequent_books_s.support[1:11],color='rgmyk')
plt.xlabel('item-sets');plt.ylabel('support')

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