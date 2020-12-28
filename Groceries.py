# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:17:01 2020

@author: shara
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
grocery = []
with open("F:/Warun/DS Assignments/DS Assignments/Association Rules/groceries.csv") as F:
    grocery = F.read()
grocery = grocery.split("\n")
grocery_list = []
for i in grocery:
    grocery_list.append(i.split(","))
all_grocery_list = [i for item in grocery_list for i in item]

from collections import Counter, OrderedDict
item_frequencies = Counter(all_grocery_list)
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1]) 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))
groceries_series = pd.DataFrame(pd.Series(grocery_list))
groceries_series = groceries_series.iloc[:9835, :]
groceries_series.columns = ["transaction"]
x = groceries_series['transaction'].str.join(sep = '*').str.get_dummies(sep = '*')


frequent_itemsets = apriori(x,min_support = 0.005, max_len = 3, use_colnames = True, low_memory=(True))
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
Rules = association_rules(frequent_itemsets, metric= 'lift', min_threshold= 1)

frequent_itemsets1 = apriori(x,min_support = 0.01, max_len = 4, use_colnames = True, low_memory=(True))
Rules1 = association_rules(frequent_itemsets1, metric= 'lift', min_threshold= 1)

frequent_itemsets2 = apriori(x,min_support = 0.008, max_len = 3, use_colnames = True, low_memory=(True))
Rules2 = association_rules(frequent_itemsets2, metric= 'lift', min_threshold= 1)

frequent_itemsets3 = apriori(x,min_support = 0.01, max_len = 3, use_colnames = True, low_memory=(True))
Rules3 = association_rules(frequent_itemsets2, metric= 'confidence', min_threshold= 0.3)

frequent_itemsets4 = apriori(x,min_support = 0.06, max_len = 3, use_colnames = True, low_memory=(True))
Rules4 = association_rules(frequent_itemsets2, metric= 'lift', min_threshold= 1.4)

Rules.head(20)
Rules.sort_values('lift', ascending = False).head(20)
Rules_r = Rules.sort_values('lift', ascending = False)

plt.scatter(Rules['support'], Rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(Rules["support"], Rules["lift"], alpha=0.5)
plt.xlabel("support")
plt.ylabel("lift")
plt.title("Support vs Lift")
plt.show()


plt.scatter(Rules["confidence"], Rules["lift"], alpha=0.5)
plt.xlabel("confidence")
plt.ylabel("lift")
plt.title("confidence vs Lift")
plt.show()


fit = np.polyfit(Rules['lift'], Rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(Rules['lift'], Rules['confidence'], 'yo', Rules['lift'], fit_fn(Rules['lift']))
