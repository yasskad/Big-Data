import os
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
from pyspark.mllib.feature import HashingTF, IDF


def find_outliers(column):
    """
    Input: sequence of text data
    Output: outlier values
    """
    column_2 = np.array(column.collect())
    max_len = 0
    for i in range(column_2.shape[0]):
        max_len = max(max_len, len(column_2[i].split()))
    #SHORT: categories + solving conflicts with edit
    if max_len <= 3:
        outliers = short_text_outliers(column_2)
    #LONG: tf-idf (fit gaussian on k-nearest or say we need to find some nb of outliers)
    else:
        outliers = long_text_outliers(column)
    
    return outliers
    
    
def edit_distance(str1, str2):
    len_1, len_2, dist =len(str1)+1, len(str2)+1, {}
    for i in range(len_1): 
        dist[i,0]=i
    for j in range(len_2): 
        dist[0,j]=j
    for i in range(1, len_1):
        for j in range(1, len_2):
            cost = 0 if str1[i-1] == str2[j-1] else 1
            dist[i,j] = min(dist[i, j-1]+1, dist[i-1, j]+1, dist[i-1, j-1]+cost)
    return dist[i,j]


def is_matching(categories, category):
    for current in categories[np.where(categories!=category)[0]]:
        len1, len2 = len(current), len(category)
        if edit_distance(current, category) <= max(len1, len2)/5:
            return True, current
    return False, ""



def short_text_outliers(column):
    categories_f = np.unique(column).copy()
    categ_sizes = []
    for category in categories_f:
        categ_sizes.append(len(np.where(column==category)[0]))
    percent_size = np.percentile(categ_sizes, 15)
    
    # first checks if low pop categories are due to typing errors
    categories = categories_f.copy()
    all_true = True
    while all_true:
        all_true = True
        percent_size = percent_size * 2
        for current in categories_f:
            if len(np.where(column == current)[0]) < percent_size:
                is_matched, match_categ = is_matching(categories, current)
                if is_matched:
                    column[np.where(column == current)[0]] = match_categ
                    categories = np.unique(column)
            else:
                all_true = False
    # Then, we return outliers (low population)
    outliers = []
    for current in categories:
        if len(np.where(column == current)[0]) <= percent_size:
            outliers = outliers + [current] 
    return outliers



def long_text_outliers(column):
    k = 10
    #vectorizer = TfidfVectorizer(max_features=250, use_idf=True)
    #data_mat = vectorizer.fit_transform(column).toarray()
    seq = column.map(lambda line: line.split(" "))

    hashingTF = HashingTF()
    tf = hashingTF.transform(seq)
    tf.cache()
    idf = IDF(minDocFreq=2).fit(tf)
    tfidf = idf.transform(tf)
    
    data_mat = []
    for i in range(len(tfidf.collect())):
        data_mat = np.hstack((data_mat, tfidf.collect()[i].values))
        

    avg_k_dist = []
    for i in range(len(data_mat)):
        dist = np.linalg.norm(data_mat - data_mat[i], 2, axis=1)
        avg_k_dist.append(1/k*np.sum(np.sort(dist)[1:(k+1)]))
    
    avg_k_dist = np.array(avg_k_dist)
    mean, std = np.mean(avg_k_dist), np.std(avg_k_dist)
    outliers = np.where(avg_k_dist > mean + 3*std)[0]
    return list(np.unique(avg_k_dist[outliers]))



if __name__=='__main__':
    print("")
