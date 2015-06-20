__author__ = 'Gorilla'

import math

"""
Calculation of Mean + Standard Deviation
"""
def calculate_mean_sd(a):
    mean = math.fsum(a) / len(a)
    sum = 0
    for i in a:
        sum += (i-mean)**2

    print "SSQD = ", sum

    sd = (sum/len(a))**0.5

    return mean, sd

def possible_mean(L):
    return sum(L)/len(L)

def possible_variance(L):
    mu = possible_mean(L)
    temp = 0
    for e in L:
        temp += (e-mu)**2
    return float(temp) / len(L)


"""
Note: Chi-Square is not implemented here, it needs a more robust tool like Excel sheet to
calculate the values of expectation and observation.
"""




"""
t-test: use to evaluate the correlation between 2 variables
sigma value used in t-test: IV = nominal or ordinal, DV = interval or ratio
@:param N1, N2: the cardinality of 2 sets observed, respectively
@:param STD1, STD2: standard deviation of 2 sets observed, respectively
"""
def calculate_std_error_diff_mean(N1, STD1, N2, STD2):
    nom1    = (N1 * STD1*STD1) + (N2 * STD2*STD2)
    denom1  = N1 + N2 -2 # df value: data freedom
    print "df = ", denom1
    nom2    = N1 + N2
    denom2  = N1*N2

    return ((float(nom1)/denom1)**0.5) * ((float(nom2)/denom2)**0.5)

def calculate_t_test(m1, m2, sigma):
    return float(m1 - m2) / sigma
# END for t-test


# CALCULATE t-test
# m1, STD1 = calculate_mean_sd([17, 24, 23, 18, 15, 21, 17, 21, 17, 24])
# m2, STD2 = calculate_mean_sd([28, 30, 32, 13, 32, 28, 20, 19, 32, 15])
# n1 = 10
# n2 = 10

#EXAM
X=[19, 9, 4, 15, 16, 17, 4, 14, 6, 12, 6, 6, 11, 8, 12, 15, 10, 14]
Y=[15, 11, 14, 19, 16, 18, 11, 9, 9, 8, 14, 8, 24, 16, 12, 6]
print len(X), len(Y)
m1, STD1 = calculate_mean_sd(X)
m2, STD2 = calculate_mean_sd(Y)
n1 = 18
n2 = 16

print "mean1 =", m1, "std1 = ", STD1
print "mean2 =", m2, "std2 = ", STD2
sigma =  calculate_std_error_diff_mean(n1, STD1, n2, STD2)
print "sigma = ", sigma
print "t = ", calculate_t_test(m1, m2, sigma)



"""
Analysis of Variarion (ANOVA) - or F-test
for 2 and more groups
@:param A: a list of lists, each represent the distribution for one variable
Will output the ANOVA for overall elements of list in A
"""
def anova(A):
    # number of lists in A
    num = len(A)
    print "number of group is: ", num
    sum = [0 for i in range(num)]
    sqrSum = [0 for i in range(num)]
    mean = [0 for i in range(num)]

    for i in range(num): # 0 -> num - 1
        # calculate the sum and the squared Sum
        # for each list in A
        s, ss = 0, 0
        # deal with list a
        for a in A[i]:
            s += a
            ss += a**2
        # after iterating all elements
        # update the needed statistics
        sum[i] = s
        sqrSum[i] = ss
        mean[i] = float(s) / len(A[i])
        print "=========For group", i+1, "the sum is", sum[i], "squared sum is: ", sqrSum[i], "\tmean is: ", mean[i]

    # calculate overall mean
    overallSum = 0.0
    overallLen = 0
    for i in range(num):
        overallSum += sum[i]
        overallLen += len(A[i])
    overallMean = float(overallSum) / overallLen
    print "Overall cardinality is: ", overallLen
    print "Overall Mean is: ", overallMean

    # next, calculate the Total Sum of Square
    overallSqrSum = 0.0
    for ss in sqrSum: overallSqrSum += ss
    SSt = float(overallSqrSum - overallLen * (overallMean**2))
    print "Total Sum of Squares is (SSt): ", SSt

    # next, calculate sum of squared BETWEEN the groups in A
    # = SIGMA ( N[i] * (mean[i] - overallMean)^2)
    sumOfSqrBtw = 0.0
    for i in range(num):
        sumOfSqrBtw += float(len(A[i]) * ( mean[i] - overallMean)**2)
    print "Sum of Square Between is (SSb): ", sumOfSqrBtw
    print "Sum of Square Within is: (SSw)", SSt - sumOfSqrBtw

    # calculate DFw and DFb
    DFw = overallLen - num
    DFb = num -1
    print "DFw = ", DFw, " DFb = ", DFb

    # calculate mean of square between and mean of square within
    MSw = float(SSt - sumOfSqrBtw) / DFw
    MSb = float(sumOfSqrBtw) / DFb
    print "Mean of Square Within is (MSw) ", MSw
    print "Mean of Square Between is (MSb) ", MSb

    # Finally, we have statistics F
    # and will compared with the critical value
    # normally significant level is 5%
    F = float(MSb) / MSw
    print "F is: ", F


# a test for ANOVA
# a = [1,20,13,0,3,5,13,7,14,7]
# b = [1,15,7,10,14,5,3,12,2,6]
# c = [28,4,5,21,25,11,21,3,12,2]
# d = [13,25,9,18,10,25,21,6,24,18]
# A = [a, b, c, d]
# a=[6,16,9,19,8,14,2,19,20,19,7,11,14,10,23,4,12,2,12,18]
# b=[10,5,3,21,12,1,16,25,23,27,25,9,2,3,20,11,20,19,3,18]
# c=[15,5,19,6,5,19,18,7,23,3,2,10,11,22,24,14,8,14,12,22]
# A = [a, b, c]
# anova(A)
#

















"""
This is the implementation of the Linear Regression between an IV and a DV
@:param X: set of the independent values
@:param Y: set of the corresponding dependent values
"""
def regression(X, Y):
    assert(len(X) == len(Y))
    N = len(X)

    # y^ = alpha + beta * x
    # alpha = meanY - beta * mean X
    # beta = (N*sum(xy) - sum(x)sum(y))   /   (N*sum(x^2) - sum(x)^2))
    sumX, sumY, sumXY, sumSqrX = 0, 0, 0, 0
    for x in X: sumX += x
    for y in Y: sumY += y
    for i in range(N):
        sumXY += X[i]*Y[i]
        sumSqrX += X[i]**2
    meanX, meanY = float(sumX) / N, float(sumY) / N
    print "SumX: ", sumX, " SumY: ", sumY, " sumXY: ", \
        sumXY, " sumSqrX: ", sumSqrX, " meanX: ", meanX, " meanY: ", meanY

    # calculate beta
    beta = float((N*sumXY) - sumX*sumY) / (N*sumSqrX - sumX**2)
    #calculate alpha
    alpha = float(meanY - beta*meanX)
    print "beta: ", beta, " alpha: ", alpha, " Regression line: y^ = alpha + beta*x"

    return alpha, beta, meanX

def regression_test(X, Y):
    alpha, beta, meanX = regression(X, Y)
    N = len(X)
    df = N - 2
    print "-----df = ", df

    # now calculate alpha^
    Yhat = [alpha + beta*X[i] for i in range(N)]  # array of predicted values

    temp1 = 0
    for i in range(N): temp1 += (Y[i] - Yhat[i])**2
    print "------SUM(y - y^)^2 = ", temp1
    nominator = (float(temp1) / (N-2))**0.5

    temp2 = 0
    for i in range(N): temp2 += (X[i] - meanX)**2
    print "------SUM(x - meanX)^2 = ", temp2
    denominator = (temp2)**0.5

    sigma = float(nominator) / denominator
    print "nom: ", nominator, " denom: ", denominator, " sigma: ", sigma

    t = float(beta) / sigma
    print "t value: ", t
    return t



# Test for Regression
# X = [62,38,60,56,45,26,34,31,59,58,55,27,27,33,62,27,50,33,42,59]
# Y = [62,53,59,63,45,70,72,60,31,60,72,51,46,25,66,46,30,31,41,74]
# X = [118,100,96,95,29,60,65,26,50,41,78,101,110,59]
# Y = [27,58,45,58,54,62,67,64,65,49,48,49,33,63]
# X = [35,31,29,11,11,13,30,25,36,20,17,22]
# Y = [58,41,46,42,48,43,56,40,69,57,57,65]
# X = [9, 3, 11, 0, 10, 5, 3, 0, 2, 11, 3, 5, 4, 8]
# Y = [59, 52, 50, 71, 55, 51, 97, 77, 59, 59, 72, 78, 85, 62]
# X = [7, 55, 31, 117, 58, 110, 102, 18, 5, 92, 36, 76, 37, 12, 40]
# Y = [28, 12, 5, 8, 20, 1, 1, 29, 22, 18, 10, 10, 19, 17, 6]
# regression_test(X, Y)







"""
This is the implementation of Pearson correlation (r correlation)
IV and DV are both of interval or ratio types
It's similar to regression but the purpose is different
Not prediction anymore, now determine the strong or weak co-variance between the 2 variables X and Y
For example: as X increases, Y decreases with the rate: fast or slow
absolute(r) ~ 1 => fast, else if ~0 => slow
r ~ [-1, 1]
@:param X: list of independent values
@:param Y: list of the corresponding dependent values
"""
def r_correlation(X, Y):
    assert(len(X) == len(Y))

    N = len(X)
    print "The length of data is: ", N, "\tdf = ", N-2

    sumX, sumY = 0, 0
    for x in X: sumX += x
    for y in Y: sumY += y
    meanX, meanY = float(sumX) / N, float(sumY) / N
    print "------meanX = ", meanX, " meanY = ", meanY

    # calculate the nominator of r-correlation
    # it is the co-variance without being normalized by N
    nominator = 0
    for i in range(N): nominator += (X[i] - meanX)*(Y[i] - meanY)
    print "SUM OF THE PRODUCT OF MEAN DEVIATIONS OF X AND Y (Covariance): ", nominator


    #calculate the denominator of r-correlation
    # it is the standard variation of X times that of Y (sdX * sdY)
    sumSS_X, sumSS_Y = 0, 0
    for x in X: sumSS_X += (x-meanX)**2
    for y in Y: sumSS_Y += (y-meanY)**2
    denominator = (sumSS_X * sumSS_Y)**0.5
    print "Sum of squared deviation: for X = ", sumSS_X, " for Y = ", sumSS_Y
    print "demon: ", denominator

    r = float(nominator) / denominator
    print "r = ", r
    return r

def t_test_correlation(X, Y):
    r = r_correlation(X, Y)

    N = len(X)

    t = r * ((N-2)**0.5) / (1 - r**2)**0.5
    print "t = ", t


# Test for r-correlation
# X = [2.9,3.9,2.9,7.1,6.9,7,7.4,7.7,4.8,2.2,6.5,5.8,5.9,6.4,4.3,5.7,5.3,5.5,4.9,6.8]
# Y = [1,0,5,7,5,10,5,6,0,7,6,8,6,6,9,10,7,3,8,4]
# X = [5,7,7,3,7,2,6,2,3,2,1,5,0,5,2,2]
# Y = [25,12,14,17,16,23,17,21,17,24,23,18,29,11,14,20]
# X = [29,54,15,18,80,71,87,65,37,41,76,60,27,97,82,63,4,81,95,101]
# Y = [19,30,27,15,5,19,15,6,19,22,11,9,19,12,11,12,26,11,18,11]
# X = [16,77,9,16,80,69,50,78,52,47,79,66,11,65]
# Y = [0,15,15,3,2,22,28,20,17,10,12,27,21,13]
# X = [8, 27, 4, 1, 56, 47, 24, 18, 2, 7, 48, 24, 25, 16, 2]
# Y = [72, 49, 83, 54, 99, 82, 97, 55, 57, 75, 82, 66, 65, 76, 49]
# X = [7, 55, 31, 117, 58, 110, 102, 18, 5, 92, 36, 76, 37, 12, 40]
# Y = [28, 12, 5, 8, 20, 1, 1, 29, 22, 18, 10, 10, 19, 17, 6]
# t_test_correlation(X, Y)
#

