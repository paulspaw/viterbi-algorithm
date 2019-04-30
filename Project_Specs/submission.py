#!/usr/bin/env python3
# coding=UTF-8
'''
@Description: 
@Author: Peng LIU, ZhiHao LI
@LastEditors: Peng LIU
@Date: 2019-03-29 23:14:10
@LastEditTime: 2019-05-01 02:37:40
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
        N = int(file.readline())
        stateSet = {}
        matrixA = np.zeros((N, N))
        pi = [0 for i in range(N)]
        end = [0 for i in range(N)]
        
        ID = 0
        while ID < N:
            stateName = file.readline().strip()
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
        M = int(file.readline())
        symbolSet = {}
        matrixB = np.zeros((N, M+1))

        ID = 0
        while ID < M:
            symbol = file.readline().strip()
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


# In[323]:


def query_to_token(line, symbolSet): 
    tokens = re.findall(r"[A-Za-z0-9.]+|[,|\.|/|;|\'|`|\[|\]|<|>|\?|:|\"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|\-|=|\_|\+]", line)
    Obs = [0 for i in range(len(tokens))]
    for i in range(len(tokens)):
        if tokens[i] in symbolSet.keys():
            Obs[i] = symbolSet[tokens[i]]
        else:
            Obs[i] = symbolSet["UNK"]
    # print(Obs)
    return Obs


def viterbi(N,Obs,PI,END,A,B):
    path = []
    T = len(Obs)
    delta = np.zeros((N, T))
    record = np.zeros((N, T), int)
    psi = [[[]] * T for i in range(N)]

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


def viterbi_algorithm(State_File, Symbol_File, Query_File):
    N, stateSet, A, PI, END = StateFileProcessing(State_File,Smooth=1)
    symbolSet, B = SymbolFileProcessing(N, Symbol_File, Smooth=1)

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

def top_k(N,Obs,PI,END,A,B,K):
    
    T = len(Obs)
    
    delta = np.zeros((N, T, K), float)
    record = np.zeros((N, T, K), int)
    
    for state in range(N):
<<<<<<< HEAD
        delta[state, 0, K] = PI[state] * B[state][Obs[0]] 
        record[state, 0, K] = state
=======
        delta[state, 0, 0] = PI[state] * B[state][Obs[0]] 
        # record[state, 0, 0] = state
        
        # for k in range(1, K):
        #     delta[state, 0, k] = 0.0
        #     record[state, 0, k] = state
>>>>>>> 20925e9d838b8470715ece4c38b909fefd642436
        
    for ts in range(1, T):
        for sn in range(N):
            prob_state = []
            for sp in range(N):
                for k in range(K):
                    prob = delta[sp, ts-1, k] * A[sp, sn] * B[sn, Obs[ts]]
                    state = sp
                    prob_state.append((prob, state))
            prob_state_sorted = sorted(prob_state, key=lambda x: x[0], reverse=True)
            
            for k in range(K):
                delta[sn, ts, k] = prob_state_sorted[k][0]
                record[sn, ts, k] = prob_state_sorted[k][1]
            # print(record[sn,ts,1])
            
    prob_state = []
    for state in range(N):
        for k in range(K):
            prob = delta[state, T-1, k]
            prob_state.append((prob, state))
            
    prob_state_sorted = sorted(prob_state, key=lambda x: x[0], reverse=True)
    
    path = [[0 for i in range(T+1)] for j in range(K)]
    
    #### 这一部分 回溯出了问题
    for k in range(K):
        maxProb = prob_state_sorted[k][0]
        maxIndex = prob_state_sorted[k][1]
        
        path[k][-1] = maxProb
        path[k][-2] = maxIndex
<<<<<<< HEAD

        for t in range(T-2, -1, -1):
            new_state = record[path[k][t+1]][k][t+1]
            path[k][t] = new_state
        # col = -1
       
        # while True:
        #     if T <= -col:
        #         break
        #     maxState = record[maxIndex][k][col]
        #     maxIndex = maxState
        #     col -= 1
        #     path[k][col-1] = maxState
        # maxProb = np.log(maxProb * END[path[k][-2]])
        # path[k][-1] = maxProb
    # print(path)
    print(delta)
    print()
        
=======
        loc = -2
        col = -3
        while True:
            if T <= -col:
                break
            preState = record[maxIndex][col][k]
            path[k][loc - 1] = preState
            maxIndex = preState
            col -= 1
            loc -= 1      
        maxProb = np.log(maxProb * END[path[k][-2]])
        path[k][-1] = maxProb

>>>>>>> 20925e9d838b8470715ece4c38b909fefd642436
    return path

def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    N, stateSet, A, PI, END = StateFileProcessing(State_File,Smooth=1)
    symbolSet, B = SymbolFileProcessing(N, Symbol_File, Smooth=1)
    results = []

    with open(Query_File, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            
            Obs = query_to_token(line, symbolSet)
            result = top_k(N,Obs,PI,END,A,B,k)

            for sub in result:
                sub.insert(0, stateSet["BEGIN"])
                sub.insert(-1, stateSet["END"]) 
                results.append(sub)
            break
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
<<<<<<< HEAD
    import time;  # 引入time模块

    State_File ='./toy_example/State_File'
    Symbol_File='./toy_example/Symbol_File'
    Query_File ='./toy_example/Query_File'
 
    # State_File ='./dev_set/State_File'
    # Symbol_File='./dev_set/Symbol_File'
    # Query_File ='./dev_set/Query_File'
    ticks = time.time()
=======
    # import time;  # 引入time模块
 
    State_File ='./toy_example/State_File'
    Symbol_File='./toy_example/Symbol_File'
    Query_File ='./toy_example/Query_File'
    # ticks = time.time()
>>>>>>> 20925e9d838b8470715ece4c38b909fefd642436
    # viterbi_result1 = viterbi_algorithm(State_File, Symbol_File, Query_File)
    viterbi_result2 = top_k_viterbi(State_File, Symbol_File, Query_File, k=2)
    # viterbi_result3 = advanced_decoding(State_File, Symbol_File, Query_File)
    # ticks2 = time.time()
    # print(ticks2 - ticks)
<<<<<<< HEAD
    # for row in viterbi_result:
=======
    # for row in viterbi_result2:
>>>>>>> 20925e9d838b8470715ece4c38b909fefd642436
    #     print(row)





