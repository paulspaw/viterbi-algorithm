import json
import re
import numpy as np

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

def SymbolFileProcessing(Symbol_File, Smooth):
    with open(Symbol_File,'r') as file:
        M = int(file.readline())
        symbolSet = {}
        matrixB = np.zeros((M+2, M+1))

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
            
        for i in range(0, M):
            total = matrixB[i].sum()
            for j in range(0, M+1):
                if j == ID or matrixB[i][j] == 0:
                    matrixB[i][j] = 1 / (total + M + 1)
                else:
                    matrixB[i][j] = (matrixB[i][j] + (1 * Smooth)) / (total + M * Smooth + 1)
        
    file.close()
    return symbolSet, matrixB

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
    path = [maxIndex]
    col = -1
    while True:
        if T <= -col:
            break
        maxState = record[maxIndex][col]
        maxIndex = maxState
        path.append(maxState)
        col -= 1
    path = list(reversed(path))
    path.append(round(np.log(maxProb),6))
    
    return path

def top_k(N,Obs,PI,END,A,B,K):
    path = []
    T = len(Obs)
    delta = np.zeros((N, T))
    deltaCopy = np.zeros((N, T))
    record = np.zeros((N, T), int)
    psi = [[[]] * T for i in range(N)]

    delta[:, 0] = PI * B[:, Obs[0]]
    deltaCopy[:, 0] = PI * B[:, Obs[0]]
    for ts in range(1, T):      
        for sn in range(N):     
            for sp in range(N):  
                prob = delta[sp][ts-1] * A[sp][sn] * B[sn][Obs[ts]]
                if prob > delta[sn][ts]:
                    delta[sn][ts] = prob
                    deltaCopy[sn][ts] = prob
                    record[sn][ts] = sp
    delta[:, -1] = END * delta[:, -1]
    deltaCopy[:, -1] = END * deltaCopy[:, -1]
    
    maxProb = []
    maxIndex = []
    tempK = K
    while tempK > 0:
        tempK -= 1
        maxTempProb = 0
        maxTempIndex = 0
        for index in range(len(delta)):
            if len(maxIndex):
                if index == maxIndex[-1]:
                    continue
            if delta[index][-1] > maxTempProb:
                maxTempProb = delta[index][-1]
                maxTempIndex = index
            elif delta[index][-1] == maxTempProb:
                for i in range(1, T):
                    if delta[index][-1-i] > maxTempProb:
                        maxTempProb = delta[index][-1]
                        maxTempIndex = index
        maxProb.append(maxTempProb)
        maxIndex.append(maxTempIndex)
    
    pathes = []
    for index in range(len(maxIndex)):
        maxTempIndex = maxIndex[index]
        path = [maxTempIndex]
        col = -1
        while True:
            if T <= -col:
                break
            maxTempState = record[maxTempIndex][col]
            maxTempIndex = maxTempState
            path.append(maxTempState)
            col -= 1
        path = list(reversed(path))
        path.append(round(np.log(maxProb[index]),6))
        pathes.append(path) 
        
    return T, pathes

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File):
    N, stateSet, A, PI, END = StateFileProcessing(State_File,Smooth=1)
    symbolSet, B = SymbolFileProcessing(Symbol_File, Smooth=1)

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

# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    N, stateSet, A, PI, END = StateFileProcessing(State_File,Smooth=1)
    symbolSet, B = SymbolFileProcessing(Symbol_File, Smooth=1)
    results = [[]for i in range(k)]
    
    with open(Query_File, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            
            Obs = query_to_token(line, symbolSet)
            t, result = top_k(N,Obs,PI,END,A,B,k)
            
            for index in range(len(result)):
                result[index].insert(0, stateSet["BEGIN"])
                result[index].insert(-1, stateSet["END"]) 
                results[index].append(result[index])
    file.close()

    return results


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...


if __name__ == "__main__":
    State_File ='./toy_example/State_File'
    Symbol_File='./toy_example/Symbol_File'
    Query_File ='./toy_example/Query_File'
    # viterbi_result = viterbi_algorithm(State_File, Symbol_File, Query_File)
    viterbi_result = top_k_viterbi(State_File, Symbol_File, Query_File, k=2)