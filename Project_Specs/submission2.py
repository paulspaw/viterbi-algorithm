import json
import re
import numpy as np

def StateFileProcessing(State_File,Smooth):
    with open (State_File,'r') as file:
        N = int(file.readline())
        stateSet = {}
        matrixA = np.zeros((N, N))
        pi = [0 for i in range(N)]
        
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
        
    file.close()
    return N, stateSet, matrixA, pi

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

def viterbi(N,Obs,PI,A,B):
    path = []
    T = len(Obs)
    delta = np.zeros((N, T))
    record = np.zeros((N, T), int)
    psi = [[[]] * T for i in range(N)]

    delta[:, 0] = PI * B[:, Obs[0]]
    
    #### 明天看看这里 还有前面的PI赋值
    end = [0 for i in range(N)]
    for i in range(N):
        end[i] = A[i][-1]
    for ts in range(1, T):       #  timeStamp
        for sn in range(N):     #  stateNext
            for sp in range(N):  #  statePrev
                prob = delta[sp][ts-1] * A[sp][sn] * B[sn][Obs[ts]]
                if prob > delta[sn][ts]:
                    delta[sn][ts] = prob
                    record[sn][ts] = sp
    # print(psi)
    maxProb = 0
    maxIndex = 0
    for index in range(len(delta)):
        if delta[index][-1] > maxProb:
            #### 最后要乘stateEnd的概率，每个s转移到end的概率都不一样
            #### 同理，begin也是，begin到每个s的概率都不一样
            #### 最后输出概率应该是结合begin end 的概率的乘积才对
            maxProb = delta[index][-1] * end[index]
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
    path.append(np.log(maxProb))
    
    return path

def viterbi_algorithm(State_File, Symbol_File, Query_File):
    N, stateSet, A, PI = StateFileProcessing(State_File,Smooth=1)
    symbolSet, B = SymbolFileProcessing(Symbol_File, Smooth=1)

    results = []
    with open(Query_File, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            
            Obs = query_to_token(line, symbolSet)
            result = viterbi(N,Obs,PI,A,B)
            result.insert(0, stateSet["BEGIN"])
            result.insert(-1, stateSet["END"])
            results.append(result)
            print(result)
    file.close()
    return results


if __name__ == "__main__":
    State_File ='./toy_example/State_File'
    Symbol_File='./toy_example/Symbol_File'
    Query_File ='./toy_example/Query_File'
    viterbi_result = viterbi_algorithm(State_File, Symbol_File, Query_File)