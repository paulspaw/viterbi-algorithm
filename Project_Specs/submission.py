#!/usr/bin/env python3
# coding=UTF-8
'''
@Description: 
@Author: Peng LIU, ZhiHao LI
@LastEditors: Peng LIU
@Date: 2019-03-29 23:14:10
@LastEditTime: 2019-04-27 16:43:22
'''

# Import your files here...
import json
import re
import numpy as np

'''
    N：隐藏状态数 hidden states

    M：观测状态数 observed states

    A：状态转移矩阵 transition matrix

    B：发射矩阵  emission matrix
    
    pi：初始隐状态向量 initial state vector
'''

# deal with state_file
def StateFileProcessing(State_File,Smooth):
    with open (State_File,'r') as file:
        N = int(file.readline())              # integer N , which is the number of states
        state_set = dict()                    # store the set of state
        transition_prob = dict()              # store transition probability  
        state_prob = dict()                   # store state initialising probability
        ID = 0                                # ID of states
        #count = 0                             # number of transitions

        # Scan descriptive name of the states.
        while ID < N:
                state_name = file.readline().strip()
                state_set[state_name] = ID
                ID += 1
                
        # Scan the transitions and weight.
        while True:
            line = file.readline()
            if not line:
                break
            items = line.split()

            state1 = int(items[0])      # The state before the transition
            state2 = int(items[1])      # The state after the transition
            weight = int(items[2])      # The weight of the transition

            transition_prob.setdefault(state1,{})[state2] = weight
            
        
        # Convert weight into probability.
        for keys,values in transition_prob.items():
            total = 0
            for value in values.values():
                total += value
            # Scan each state in state_set.
            for state in state_set.values():
                # Case 1: state is already existing
                if state in values.keys():
                    transition_prob[keys][state] = (transition_prob[keys][state] + Smooth)/(total + (N-1) * Smooth)
                # Case 2: state is not existing
                else:
                    if state == state_set['BEGIN']:
                        # For the BEGIN state, there is no transition to it, i.e., the probability is indeed 0.0.
                        transition_prob.setdefault(keys,{})[state] = 0.0
                    else:
                        transition_prob.setdefault(keys,{})[state] = (1 * Smooth)/(total + (N-1) * Smooth)
                        
        # For the END states, there is no transition from it, i.e., the probability is indeed 0.0.
        for state in state_set.values():
            transition_prob.setdefault(state_set['END'],{})[state] = 0.0
            # Initialize state probability
            state_prob[state] = 1/N
                        
    file.close()
    return N, state_set, transition_prob, state_prob 

# deal with symbol file 
def SymbolFileProcessing(Symbol_File, state_set,Smooth):
    with open(Symbol_File,'r') as file:
        M = int(file.readline())        # integer M, which is the number of symbols. M个元素
        symbol_set = dict()             # store the set of symbol 元素的种类
        emission_prob = dict()          # store emission probability    状态 - 种类 - 数量    
        ID = 0    
        
        # Scan descriptive name of the symbols.
        while ID < M:
            symbol = file.readline().strip()
            symbol_set[symbol] = ID
            ID += 1

        # Scan the frequency of emissions.
        while True:
            line = file.readline()
            if not line:
                break
            items = line.split()
            
            state = int(items[0])      # The state
            ele_type = int(items[1])      # The type of the element
            amount = int(items[2])      # The quantity of the element

            emission_prob.setdefault(state,{})[ele_type] = amount

        # Convert frequency into probability.
        for keys,values in emission_prob.items():
            total = 0
            for value in values.values():
                total += value
            # Scan each symbol in symbol_set.
            for symbol in symbol_set.values():
                # Case 1: symbol is already existing
                if symbol in values.keys():
                    emission_prob[keys][symbol] = (emission_prob[keys][symbol]+(1 * Smooth))/(total+M*Smooth+1)
                # Case 2: symbol is not existing
                else:   
                    # ⚠️SMOOTH 部分需要调整                  
                    emission_prob.setdefault(keys,{})[symbol] = 1/(total+M+1)
            # "UNK" symbol 
            emission_prob.setdefault(keys,{})[M] = 1/(total+M+1)    # ⚠️need to be fixed
                                      
    file.close()
    return M, symbol_set, emission_prob #元素的数量，元素的名称，元素的Emission Probabilities

# 处理query的每一行
def query_to_token(line): 
    tokens = re.findall(r"[A-Za-z0-9.]+|[,|\.|/|;|\'|`|\[|\]|<|>|\?|:|\"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|\-|=|\_|\+]", line)
    return tokens

# 设计viterbi内置算法
def viterbi(N,Obs,PI,MatrixA,MatrixB):
    ROW = N
    COL = len(Obs)
    # initialized max probility matrix
    max_prob_matrix = np.zeros((ROW,COL), float)

    # backtracking matrix -- 用来记录到达t时刻的路径 -- list
    backtrack = [[[]] * COL for i in range(ROW)]

    print(max_prob_matrix)
    # Step 1: Initialize local states when t=0.
    # # 初始状态 PI * 第一个盒子的 
    # max_prob_matrix[:,0] = PI * MatrixB[:,Obs[0]]
    
    # 矩阵形式检验
    for i in backtrack:
        print(i)
   
    return MatrixB,max_prob_matrix

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    # assume smooth = 1
    # N -- 有多少个状态
    # state_set -- 状态集合 
    # transition_prob -- 转移矩阵
    # state_prob -- 初始状态概率值 π (暂时假定状态均匀分布)
    N,state_set,transition_prob,state_prob = StateFileProcessing(State_File,Smooth=1)
    
    # M -- 有多少个观测值
    # symbol_set -- 观测值集合
    # emission_prob -- 状态释放观测值的矩阵
    M, symbol_set, emission_prob = SymbolFileProcessing(Symbol_File,state_set,Smooth=1)
    # deal with Query File.
    with open(Query_File, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            # 每一行进行query_to_token处理
            token = query_to_token(line)      

            # Generate observations and initialized state probabiltiy.
            # M 最大值为UNK,假设所有为UNK
            Obs = [M for i in range(len(token))]
            #替换相应的值为对应数字
            for i in range(len(token)):
                if token[i] in symbol_set.keys():
                    Obs[i] = symbol_set[token[i]]
            # initialized transition matrix A
            MatrixA = np.zeros((N,N))
            # initialized emission matrix B
            MatrixB = np.zeros((N,M+1))
            # initialized state distribution
            PI = [0 for i in range(N)]

            for i in range(N):
                PI[i] = state_prob[i]  
                for j in range(N):
                    MatrixA[i][j] = transition_prob[i][j]
                for k in range(M+1):
                    if i < N-2:
                        MatrixB[i][k] = emission_prob[i][k]
                    else:
                        MatrixB[i][k] = 0.0
            # claculate path and maximum probility
            path, max_prob_path = viterbi(N,Obs,PI,MatrixA,MatrixB)
            print("-----------------------------")
            break
    return path,max_prob_path

# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    pass # Replace this line with your implementation...


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...

if __name__ == "__main__":
    State_File ='./toy_example/State_File'
    Symbol_File='./toy_example/Symbol_File'
    Query_File ='./toy_example/Query_File'
    #N,state_set,transition_prob,state_prob = StateFileProcessing(State_File,Smooth=1)
    #M, symbol_set, emission_prob = SymbolFileProcessing(Symbol_File,state_set,Smooth=1)
    # token = query_to_token("8/23-35 Bar%ker St., Kings'ford, NSW&= 2032")
    result = viterbi_algorithm(State_File, Symbol_File, Query_File)
    print(result)