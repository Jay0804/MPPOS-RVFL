from __future__ import division
import datetime
from math import exp
import random
from numpy import *
import numpy as np
import pandas as pd



def sig(tData, Iw, bias, num):
    '''
    tData:sample matrix:[Number of samples * Number of features]
    Iw: weight between input and hidden layer : [Number of hidden neurons * Number of features]
    bias: bias between input and hidden layer
    '''
    v = tData * Iw.T 
    bias_1 = ones((num, 1)) * bias
    v = v + bias_1
    H = 1. / (1 + exp(-v))
    return H


def MPPOS_RVFL(trainData, testData):
    firstTrainData = []
    firstTrainLabel = []
    firstC = []
    newTrainData = []
    newTrainLabel = []
    newC = []
    while 1:
        np.random.shuffle(trainData)
        flag = 0
        for t in range(0, NO):
            if trainData.T[0][t] == 1:
                flag = 1
        if flag == 1:
            break
    for t in range(0, len(trainData)):
        data = []
        if t < NO:  # Offline training data acquisition
            firstTrainLabel.append(trainData.T[0][t])
            firstC.append(trainData.T[1][t])
            for i in range(2, len(trainData.T)):
                data.append(trainData[t][i])
            firstTrainData.append(data)
            continue

        elif t == NO:  # Offline training begin
            newTrainLabel.append(trainData.T[0][t])
            newC.append(trainData.T[1][t])
            for i in range(2, len(trainData.T)):
                data.append(trainData[t][i])
            newTrainData.append(data)
            S = np.zeros((anchor, NO))
            p0 = mat(firstTrainData)
            T0 = np.zeros((NO, 2))
            for i in range(0, NO):
                a = int(firstTrainLabel[i])
                T0[i][a - 1] = 1
            C = np.eye(NO)

            Iw = mat(random.rand(nHiddenNeurons, nInputNeurons) * 2 - 1)  # Randomly generate a random matrix between the interval [-1,1]
            bias = mat(random.rand(1, nHiddenNeurons))
            H0 = sig(p0, Iw, bias, NO) 
            H0 = np.array(H0)  # H0 is the output of hidden layer
            p0 = np.array(p0)  # p0 is the direct output matrix 
            E0 = np.concatenate([H0, p0], axis=1)  # E0 is the total output matrix

            X0 = p0
            Y0 = np.array(T0)

            rand_arr = np.arange(E0.shape[0])

            while 1:  # ensuring at least 1 minority samples is included in the anchor points set
                flag_1 = 0
                flag_2 = 0
                np.random.shuffle(rand_arr)
                for k in range(0, anchor):
                    if firstTrainLabel[rand_arr[k]] == 1:
                        flag_1 += 1
                    else:
                        flag_2 += 1
                if (flag_1 != 0) and (flag_2 != 0):
                    break
            rand_arr = rand_arr[0:anchor]

            Ef = E0[rand_arr]
            Xf = X0[rand_arr]
            Yf = trainData.T[0][rand_arr]
            Tf = np.zeros((anchor, 2))
            for i in range(0, anchor):
                b = int(Yf[i])
                Tf[i][b - 1] = 1

            for i in range(0, anchor):
                for j in range(0, NO):
                    S[i][j] = np.exp((-1) * (np.sum((Xf[i] - X0[j]) ** 2) / (2 * sigma)))
            S = normalized(S)
            rho = E0.T.dot(C).dot(E0) + lam * np.eye(nHiddenNeurons + nInputNeurons) + gamma * (E0 - S.T.dot(Ef)).T.dot(
                E0 - S.T.dot(Ef))
            mu = E0.T.dot(C).dot(Y0)
            beta = np.linalg.inv(rho).dot(mu)

        elif (t > NO) and (t % New < New - 1):
            newTrainLabel.append(trainData.T[0][t])
            newC.append(trainData.T[1][t])
            for i in range(2, len(trainData.T)):
                data.append(trainData[t][i])
            newTrainData.append(data)
            continue

        else:
            newTrainLabel.append(trainData.T[0][t])
            newC.append(trainData.T[1][t])
            for i in range(2, len(trainData.T)):
                data.append(trainData[t][i])
            newTrainData.append(data)
            # Update
            num_unlab = New
            S_new = np.zeros((anchor, New))
            P_new = np.zeros((anchor, num_unlab))  # P is label probability transition matrix
            p = mat(newTrainData)
            T = np.zeros((New, 2))
            C_new = np.zeros((New, New))
            for i in range(0, New):
                b = int(newTrainLabel[i])
                T[i][b - 1] = 1
            X = np.array(p)
            Y = newTrainLabel
            Y_U_pred = np.zeros(num_unlab)

            for i in range(0, anchor):  # Calculate S_t
                for j in range(0, New):
                    S_new[i][j] = np.exp((-1) * np.sum((Xf[i] - X[j]) ** 2) / (2 * sigma))
            S_new = normalized(S_new)

            for i in range(0, anchor):  # P is label probability transition matrix
                for j in range(0, num_unlab):
                    if np.sum(S_new.T[j]) != 0:
                        P_new[i][j] = S_new[i][j] / np.sum(S_new.T[j])
            f_U = P_new.T.dot(Tf)  # f is the pseudo-label probability matrix
            for i in range(0, num_unlab):  # y is the most likely pseudo-label vector
                Y_U_pred[i] = np.argmax(f_U[i]) + 1

            m = 0
            count_minor = 0
            for i in range(0, New):  # Pseudo-labeling minority samples first
                if newC[i] == 0:
                    if Y_U_pred[m] == 1:
                        Y[i] = Y_U_pred[m]
                        C_new[i][i] = 1
                        newC[i] = 1
                        count_minor += 1
                    m += 1
                else:
                    C_new[i][i] = 1

            m = 0
            for i in range(0, New):  # Pseudo-labeling equal amount of majority samples
                if count_minor <= 0:
                    break
                if newC[i] == 0:
                    if Y_U_pred[m] == 2:
                        Y[i] = Y_U_pred[m]
                        C_new[i][i] = 1
                        newC[i] = 1
                        count_minor -= 1
                    m += 1

            for i in range(0, New):
                if newC[i] == 0:
                    Y[i] = 0

            for i in range(0, New):  # T is Y one-hot encoding
                b = int(Y[i])
                T[i][b - 1] = 1

            H = sig(p, Iw, bias, New)
            H = np.array(H)
            E = np.concatenate([H, p], axis=1)  # E is the total output

            rho = rho + E.T.dot(C_new).dot(E) + gamma * (E - S_new.T.dot(Ef)).T.dot(E - S_new.T.dot(Ef))
            mu = mu + E.T.dot(C_new).dot(T)
            beta = np.linalg.inv(rho).dot(mu)
            newTrainData = []
            newTrainLabel = []
            newC = []
    # Calculate training accuracy
    correct = 0
    sum = 0
    minor_correct_train = 0
    major_correct_train = 0
    minor_sum_train = 0
    major_sum_train = 0

    for t in range(0, len(trainData)):
        data = []
        y = trainData.T[0][t]
        for i in range(2, len(trainData.T)):
            data.append(trainData[t][i])
        p = mat(data)
        HTrain = sig(p, Iw, bias, 1)
        p = np.array(p)
        HTrain = np.array(HTrain)
        HTrain = np.concatenate([HTrain, p], axis=1)
        Y = np.dot(HTrain, beta)

        if argmax(Y) + 1 == y:
            correct += 1
            if y == 1:
                minor_correct_train += 1
            else:
                major_correct_train += 1

        sum += 1
        if y == 1:
            minor_sum_train += 1
        else:
            major_sum_train += 1
    major_acc_train = major_correct_train / major_sum_train
    minor_acc_train = minor_correct_train / minor_sum_train
    whole_acc_train = correct / sum
    # print("training accuracy is ：%f" % (correct/sum))
    # print(minor_acc_train)

    # Calculate testing accuracy
    correctTest = 0
    sumTest = 0
    minor_correct_test = 0
    major_correct_test = 0
    minor_sum_test = 0
    major_sum_test = 0

    for t in range(0, len(testData)):
        data = []
        y = testData.T[0][t]
        for i in range(2, len(testData.T)):
            data.append(testData[t][i])
        p = mat(data)
        HTrain = sig(p, Iw, bias, 1)
        p = np.array(p)
        HTrain = np.array(HTrain)
        HTrain = np.concatenate([HTrain, p], axis=1)
        Y = np.dot(HTrain, beta)
        if argmax(Y) + 1 == y:
            correctTest += 1
            if y == 1:
                minor_correct_test += 1
            else:
                major_correct_test += 1
        sumTest += 1
        if y == 1:
            minor_sum_test += 1
        else:
            major_sum_test += 1
    major_acc_test = major_correct_test / major_sum_test
    minor_acc_test = minor_correct_test / minor_sum_test
    gmean = np.sqrt(major_acc_test * minor_acc_test)
    whole_acc_test = correctTest / sumTest
    print("Test accuracy is ：%f" % (correctTest / sumTest))
    return whole_acc_test
    # return gmean
    # return minor_acc_test
    # return major_acc_test


def normalized(x):
    max = np.max(x)
    min = np.min(x)
    if max - min == 0:
        return x
    else:
        return (x - min) / (max - min)

def Ratio(data_train, data_test):
    acc_MPPOS_RVFL = 0
    for i in range(0, x):
        acc_MPPOS_RVFL += MPPOS_RVFL(data_train, data_test)
    acc_MPPOS_RVFL = acc_MPPOS_RVFL / x
    return acc_MPPOS_RVFL


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    nHiddenNeurons = 60 # Number of hidden neurons
    nInputNeurons = 52  # Number of input neurons
    NO = 400  # Number of total samples in offline training set
    New = 10  # Batch size
    x = 100  # Each test is averaged x times
    anchor = 40  # Number of anchor points
    lam = 1
    gamma = 1
    sigma = 1

    Graph_MPPOS_RVFL = np.zeros((2, 5))
    data_test = pd.read_csv('te_test_norm.csv')
    data_train = pd.read_csv('te_train_20.csv')
    data_test = np.array(data_test)
    data_train = np.array(data_train)
    Graph_MPPOS_RVFL[1][0] = Ratio(data_train, data_test)

    data_train = pd.read_csv('te_train_30.csv')
    data_train = np.array(data_train)
    Graph_MPPOS_RVFL[1][1] = Ratio(data_train, data_test)

    data_train = pd.read_csv('te_train_50.csv')
    data_train = np.array(data_train)
    Graph_MPPOS_RVFL[1][2] = Ratio(data_train, data_test)

    data_train = pd.read_csv('te_train_75.csv')
    data_train = np.array(data_train)
    Graph_MPPOS_RVFL[1][3] = Ratio(data_train, data_test)

    data_train = pd.read_csv('te_train_100.csv')
    data_train = np.array(data_train)
    Graph_MPPOS_RVFL[1][4] = Ratio(data_train, data_test)

    print('OSSRVFL-ATA-100 average')
    print(Graph_MPPOS_RVFL[1])




