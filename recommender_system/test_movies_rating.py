# coding: utf8
# recommender_system/test_movies_rating.py

import numpy as np
import recommender
from scipy.io import loadmat

data = loadmat('data/ex8_movies.mat')
# 评价矩阵
Y = data['Y']
# 是否评价矩阵
R = data['R']

movieParams = loadmat('data/ex8_movieParams.mat')
numMovies = movieParams['num_movies'][0,0]
numFeatures = movieParams['num_features'][0,0]

# 获得movies
def getMovie(line):
    return ' '.join(line.split()[1:])

with open('data/movie_ids.txt') as f:
    movieList = [getMovie(f.readline()) for i in range(numMovies)]

myRatings = np.mat(np.zeros((numMovies,1)))

myRatings[0] = 4
myRatings[97] = 2
myRatings[6] = 3
myRatings[11] = 5
myRatings[53] = 4
myRatings[63] = 5
myRatings[65] = 3
myRatings[68] = 5
myRatings[182] = 4
myRatings[225] = 5
myRatings[354] = 5
print 'New user ratings:'
for i in range(numMovies):
    if myRatings[i] > 0:
        print 'Rated %d for %s' % (myRatings[i], movieList[i])

# 训练推荐模型
Y = np.column_stack((myRatings, Y))
R = np.column_stack((myRatings, R)).astype(bool)

print '\nTraing Result:'
train, predict, getTopRecommends = recommender.getRecommender(
    Y, R, n=numFeatures, theLambda=10.0)
Theta, X = train()
rated = np.nonzero(myRatings)[0].tolist()
topRecommends = getTopRecommends(Theta, X, -1, 10, rated, movieList)

print '\nTop recommendations for you:'
for recommend in topRecommends:
    print 'Predicting rating %.1f for movie %s' % (recommend[1], recommend[0])
