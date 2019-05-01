#!/usr/bin/env python3
# coding=UTF-8
'''
@Description: 
@Author: Peng LIU, ZhiHao LI
@LastEditors: Peng LIU
@Date: 2019-03-29 23:14:10
@LastEditTime: 2019-05-01 05:28:03

'''

# Import your files here...
import re
import numpy as np

'''
    N：隐藏状态数 hidden states

    M：观测状态数 observed states

    A：状态转移矩阵 transition matrix

    B：发射矩阵  emission mat
    
    pi：初始隐状态向量 initial state vector
'''

def StateFileProcessing(State_File,Smooth):
    with open (State_File,'r') as file:
        N = int(file.readline().strip('\n'))
        stateSet = {}
        matrixA = np.zeros((N, N))
        pi = [0 for i in range(N)]
        end = [0 for i in range(N)]
        
        ID = 0
        while ID < N:
            stateName = file.readline().strip('\n').rstrip()
            stateSet[stateName] = ID
            ID += 1
            
        while True:
            line = file.readline()
            if not line:
                break
            items = line.split()
            
            statePrev = int(items[0])
            stateNext = int(items[1])
            frequency = int(items[2])
            
            matrixA[statePrev][stateNext] = frequency

        for i in range(0, N):
            if i == stateSet['END']:
                continue
            total = matrixA[i].sum()
            for j in range(0, N):
                if j == stateSet['BEGIN']:
                    continue
                matrixA[i][j] = (matrixA[i][j] + Smooth) / (total + (N - 1) * Smooth)
                
        #### PI的赋值
        for i in range(N):
            pi[i] = matrixA[stateSet['BEGIN']][i]
            end[i] = matrixA[i][-1] 
        
    file.close()
    return N, stateSet, matrixA, pi, end


def SymbolFileProcessing(N, Symbol_File, Smooth):
    with open(Symbol_File,'r') as file:
        M = int(file.readline().strip('\n'))
        symbolSet = {}
        matrixB = np.zeros((N, M+1))

        ID = 0
        while ID < M:
            symbol = file.readline().strip('\n').rstrip() 
            symbolSet[symbol] = ID
            ID += 1
        symbolSet["UNK"] = ID
        
        while True:
            line = file.readline()
            if not line:
                break
            items = line.split()
            
            state = int(items[0])
            symbol = int(items[1])
            frequency = int(items[2])
            
            matrixB[state][symbol] = frequency
            
        for i in range(0, N):
            total = matrixB[i].sum()
            for j in range(0, M+1):
                if j == ID :
                    matrixB[i][j] = 1 / (total + M + 1)
                else:
                    matrixB[i][j] = (matrixB[i][j] + (1 * Smooth)) / (total + M * Smooth + 1)
        
    file.close()
    return symbolSet, matrixB
    

def query_to_token(line,symbolSet): 
    token = re.split("([ ,()/&-])",line)
    tokens = []
    for i in range(len(token)):
        if token[i] != ' ' and token[i] != '':
            tokens.append(token[i].strip())

    # token = re.findall(r"[A-Za-z0-9.]+|[,/&()-]", line)
    Obs = [0 for i in range(len(tokens))]
    for i in range(len(tokens)):
        if tokens[i] in symbolSet.keys():
            Obs[i] = symbolSet[tokens[i]]
        else:
            Obs[i] = symbolSet["UNK"]
    
    # print(tokens)
    return Obs


def viterbi(N,Obs,PI,END,A,B):
    T = len(Obs)
    delta = np.zeros((N, T))
    record = np.zeros((N, T), int)

    delta[:, 0] = PI * B[:, Obs[0]]   
    for ts in range(1, T):       #  timeStamp
        for sn in range(N):     #  stateNext
            for sp in range(N):  #  statePrev
                prob = delta[sp][ts-1] * A[sp][sn] * B[sn][Obs[ts]]
                if prob > delta[sn][ts]:
                    delta[sn][ts] = prob
                    record[sn][ts] = sp
    # 最后要乘stateEnd的概率，每个s转移到end的概率都不一样
    # 同理，begin也是，begin到每个s的概率都不一样
    # 最后输出概率应该是结合begin end 的概率的乘积才对
    delta[:, -1] = END * delta[:, -1]

    maxProb = 0
    maxIndex = 0
    for index in range(len(delta)):
        if delta[index][-1] > maxProb:
            maxProb = delta[index][-1]
            maxIndex = index
    
    #  backtracking
    path = [0 for i in range(T+1)]
    path[-2] = maxIndex
    col = -1
    while True:
        if T <= -col:
            break
        maxState = record[maxIndex][col]
        maxIndex = maxState
        col -= 1
        path[col-1] = maxState
    path[-1] = np.log(maxProb)
    return path

def top_k(N,Obs,PI,END,A,B,K):
    if K == 1:
        return 1, 1

    T = len(Obs)
    delta = np.zeros((T, N, K))
    phi = np.zeros((T, N, K), int)
    rank = np.zeros((T, N, K), int)

    # init
    for i in range(N):
        delta[0, i, 0] = PI[i] * B[i, Obs[0]]
        phi[0, i, 0] = i

        #Set the other options to 0 initially
        for k in range(1, K):
            delta[0, i, k] = 0.0
            phi[0, i, k] = i

    #Go forward calculating top k scoring paths
    # for each state s1 from previous state s2 at time step t
    for t in range(1, T):
        for s2 in range(N):
            tmp = []
            for s1 in range(N):
                for k in range(K):
                    prob = delta[t - 1, s1, k] * A[s1, s2] * B[s2, Obs[t]]
                    state = s1
                    tmp.append((prob, state))
            tmp_sorted = sorted(tmp, key=lambda x: x[0], reverse=True)

            #We need to keep a ranking if a path crosses a state more than once
            rankDict = dict()
            for k in range(0, K):
                delta[t, s2, k] = tmp_sorted[k][0]
                phi[t, s2, k] = tmp_sorted[k][1]
                state = tmp_sorted[k][1]
                if state in rankDict:
                    rankDict[state] = rankDict[state] + 1
                else:
                    rankDict[state] = 0
                rank[t, s2, k] = rankDict[state]

    print(delta)
    print(phi)
    print(rank)
    print()

    for k in range(K):
        for i in range(N):
            delta[-1, i, k] = END[i] * delta[-1, i, k]
    
    tmp = []
    for s in range(N):
        for k in range(K):
            prob = delta[T - 1, s, k]
            tmp.append((prob, s, k))
    tmp_sorted = sorted(tmp, key=lambda x: x[0], reverse=True)

    print(tmp_sorted)

    path = [[0 for i in range(T)] for j in range(K)]
    prob = []
    for k in range(K):
        max_prob = tmp_sorted[k][0]
        state = tmp_sorted[k][1]
        rankK = tmp_sorted[k][2]
        prob.append(np.log(max_prob))
        path[k][-1] = state

        for t in range(T - 2, -1, -1):
            nextState = path[k][t+1]         # backtrack to prev state
            p = phi[t+1][nextState][rankK]   # backtrack to prev p
            path[k][t] = p
            rankK = rank[t + 1][nextState][rankK] # backtrack to prev rankk

    return path, prob

def viterbi_algorithm(State_File, Symbol_File, Query_File):
    N, stateSet, A, PI, END = StateFileProcessing(State_File,Smooth=1)
    symbolSet, B = SymbolFileProcessing(N, Symbol_File, Smooth=1)

    results = []
    with open(Query_File, 'r') as file:
        while True:
            line = file.readline().strip('\n')
            if not line:
                break
            
            Obs = query_to_token(line, symbolSet)
            result = viterbi(N,Obs,PI,END,A,B)
            result.insert(0, stateSet["BEGIN"])
            result.insert(-1, stateSet["END"])
            results.append(result)
    file.close()
    return results

def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    N, stateSet, A, PI, END = StateFileProcessing(State_File,Smooth=1)
    symbolSet, B = SymbolFileProcessing(N, Symbol_File, Smooth=1)
    results = []

    with open(Query_File, 'r') as file:
        while True:
            line = file.readline().strip('\n')
            if not line:
                break
            
            Obs = query_to_token(line, symbolSet)
            path, prob = top_k(N,Obs,PI,END,A,B,k)
            if prob == 1:
                return viterbi_algorithm(State_File, Symbol_File, Query_File)
            for i in range(len(path)):
                path[i] = [stateSet["BEGIN"]] + path[i]
                path[i].append(stateSet["END"]) 
                path[i].append(prob[i])
                results.append(path[i])
    file.close()
    return results

# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    N, stateSet, A, PI, END = StateFileProcessing(State_File,Smooth=0.01)
    symbolSet, B = SymbolFileProcessing(N, Symbol_File, Smooth=0.01)

    results = []
    with open(Query_File, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            
            Obs = query_to_token(line, symbolSet)
            result = viterbi(N,Obs,PI,END,A,B)
            result.insert(0, stateSet["BEGIN"])
            result.insert(-1, stateSet["END"])
            results.append(result)
    file.close()

    return results

if __name__ == "__main__":
    # import time;  # 引入time模块

    State_File ='./toy_example/State_File'
    Symbol_File='./toy_example/Symbol_File'
    Query_File ='./toy_example/Query_File'
    # ticks = time.time()
    # viterbi_result1 = viterbi_algorithm(State_File, Symbol_File, Query_File)
    viterbi_result2 = top_k_viterbi(State_File, Symbol_File, Query_File, k=2)
    # viterbi_result3 = advanced_decoding(State_File, Symbol_File, Query_File)
    # ticks2 = time.time()
    # print(ticks2 - ticks)
    # print(viterbi_result2)
    for row in viterbi_result2:
        print(row)