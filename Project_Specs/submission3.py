import json
import re
import numpy as np
# from __future__ import division

def read_state(State_File):
    '''
    :param State_File: file includes state set and state transition matrix
    :return N: number of states
    :return state_set: a dict contains all states' ID and name
    :return transition_prob: a dict contains transition probability 
    :return state_prob: a dict contains states and their probability
    '''
    with open(State_File, 'r') as file:
        N = int(file.readline().strip('\n'))     # read the first line to get N value
        state_set = dict()                       # store the set of state
        transition_prob = dict()                 # store transition probability  
        state_prob = dict()                      # store state initialising probability
        ID = 0                                   # ID of states
        cnt = 0                                  # number of transitions
        
        # Scan descriptive name of the states.
        while ID < N:
            state = file.readline().strip('\n').rstrip()  # one state in each line
            state_set[state] = ID
            ID = ID + 1
        
        # Scan the frequency of transitions.
        while True:
            line = file.readline()
            if not line:
                break
            items = line.split(' ')
            # Add new probability with key + value.
            transition_prob.setdefault(int(items[0]),{})[int(items[1])] = int(items[2])
            cnt = cnt + 1
        
        # Convert frequency into probability.
        for keys,values in transition_prob.items():
            total = 0
            for value in values.values():
                total = total + value
            # Scan each state in state_set.
            for state in state_set.values():
                # Case-I: state is already existing
                if state in values.keys():
#                     transition_prob[keys][state] = round((transition_prob[keys][state]+1)/(total+N-1),1)
                    transition_prob[keys][state] = (transition_prob[keys][state]+1)/(total+N-1)
                # Case-II: state is not existing
                else:
                    if state == state_set['BEGIN']:
                        transition_prob.setdefault(keys,{})[state] = 0.0
                    else:
#                         transition_prob.setdefault(keys,{})[state] = round(1/(total+N-1),1)
                        transition_prob.setdefault(keys,{})[state] = 1/(total+N-1)
            
        # Initialize state probability and Add "END" state with no outing states.
        for state in state_set.values():
            transition_prob.setdefault(state_set['END'],{})[state] = 0.0
#             state_prob[state] = round(1/N,1)
            state_prob[state] = 1/N
            
    return N, state_set, transition_prob, state_prob

def read_symbol(Symbol_File, state_set):
    '''
    :param Symbol_File: file includes symbol set and emission probability
    :param state_set: a set of state
    :return M: number of symbol
    :return symbol_set: a dict contains all symbols' ID and name
    :return emission_prob: a dict contains emission probability 
    '''
    with open(Symbol_File, 'r') as file:
        M = int(file.readline().strip('\n'))     # read the first line to get M value
        symbol_set = dict()                      # store the set of symbol
        emission_prob = dict()                   # store emission probability        
        ID = 0                                   # ID of symbols
        
        # Scan descriptive name of the symbols.
        while ID < M:
            symbol = file.readline().strip('\n').rstrip()  # one symbol in each line
#             symbol_set[ID] = symbol
            symbol_set[symbol] = ID
            ID = ID + 1
        
        # Scan the frequency of emissions.
        while True:
            line = file.readline()
            if not line:
                break
            items = line.split(' ')
            # Add new probability with key + value.
            emission_prob.setdefault(int(items[0]),{})[int(items[1])] = int(items[2])
        
        # Convert frequency into probability.
        for keys,values in emission_prob.items():
            total = 0
            for value in values.values():
                total = total + value
            # Scan each symbol in symbol_set.
            for symbol in symbol_set.values():
                # Case-I: symbol is already existing
                if symbol in values.keys():
#                     emission_prob[keys][symbol] = round((emission_prob[keys][symbol]+1)/(total+M+1),1)
                    emission_prob[keys][symbol] = (emission_prob[keys][symbol]+1)/(total+M+1)
                # Case-II: symbol is not existing
                else:
#                     emission_prob.setdefault(keys,{})[symbol] = round(1/(total+M+1),1)
                    emission_prob.setdefault(keys,{})[symbol] = 1/(total+M+1)
            # Add special symbol "UNK".
#             emission_prob.setdefault(keys,{})[M] = round(1/(total+M+1),1)
            emission_prob.setdefault(keys,{})[M] = 1/(total+M+1)
                                      
    return M, symbol_set, emission_prob

def parse_query(line):
    '''
    :param line: an address to be parsed
    :return tokens: parsed tokens sequence
    '''
    pattern = re.compile(r"[A-Za-z0-9.]+|[,&-/()]")
    tokens = pattern.findall(line)
    return tokens
def viterbi(O, Q, PI, A, B):
    '''
    :param O: observations
    :param Q: states
    :param PI: state probability
    :param A: transition probability
    :param B: emission probability
    :return path: the most possible state path
    :return prob: the largest probability  
    '''
    # Step 0: Define two matrix -- delta, psi.
    N = len(Q)
    T = len(O)
    # delta -- delta[t,i] -- 在时刻t，以状态i作为途径状态的最大的概率值是多少
    # delta[t,i] -- k个最高的概率值 == > delta[t,i,k]
    delta = np.zeros((T,N), float)     # highest probability of any path that ends at i
    # psi[t,i] -- 在时刻t，上述delta最大值的时候返回的状态是什么
    psi = np.zeros((T,N), int)         # argmax state
        
    # Step 1: Initialize local states when t=0.
    delta[0, :] = PI * B[:,O[0]]
    
    # 对应课件里的初始化工作
    for i in range(N):
        delta[0,i] = PI[i]*B[i,O[0]]

    # Step 2: Continue DP to compute local state in t = 1,3,...,T-1.
    for t in range(1, T):
        # Consider each state s2 (t) from previous state s1 (t-1)
        # t时刻，在状态s2确定的条件下，
        for s2 in range(N):
            # 遍历一次所有的状态，这些状态s1被认为是在t-1时间的结果
            for s1 in range(N):
                # 更新的过程 -- 对应课件里面的递归公式
                prob = delta[t-1, s1] * A[s1,s2] * B[s2,O[t]]
                if prob > delta[t, s2]:
                    delta[t, s2] = prob   # 记录最大概率值
                    psi[t, s2] = s1       #记录最大概率对应的状态值
    
    # Step 3: Compute the max delta value at T, which is the probability of most possible state sequence.
    # 直接计算最大的概率值作为返回信息
    max_prob = np.max(delta[T-1,:])
    
    # Step 4: Compute the most possible state at T.
    # 对应的状态值是哪个
    state_last = np.argmax(delta[T-1,:])
    
    # Step 5: Backtracking for t = T-1, T-2, ..., 1.
    path = np.zeros(T, int)         # initialize blank path
    path[-1] = state_last           # path is from tail to head
    
    for t in range(T - 2, -1, -1):
        # 在t+1时刻产生的最大的概率值对应的状态
        path[t] = psi[[t + 1], path[t + 1]]
    
    return path, np.log(max_prob)
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    '''
    :param State_File: state file
    :param Symbol_File: symbol file
    :param Query_File: query file
    '''
    
    # Generate state information.
    # N--有多少个状态
    # state_set -- 状态集合 
    # transition_prob -- 转移矩阵
    # state_prob -- 初始状态概率值 π (暂时假定状态均匀分布)
    N, state_set, transition_prob, state_prob = read_state(State_File)
    
    # Generate symbol information.    
    # M -- 有多少个观测值
    # symbol_set -- 观测值集合
    # emission_prob -- 状态释放观测值的矩阵
    M, symbol_set, emission_prob = read_symbol(Symbol_File, state_set)
    
    # Starting query.
    with open(Query_File, 'r') as file:
        while True:
            # Parse each line.
            line = file.readline()
            if not line:
                break
            query_seq = parse_query(line)      
            
            # Generate observations and initialized state probabiltiy.
            O = [M for i in range(len(query_seq))]
            for i in range(len(query_seq)):
                if query_seq[i] in symbol_set.keys():
                    O[i] = symbol_set[query_seq[i]]

            Q = range(N)                # 观测序列
            
            # Convert dict into matrix -- A and B.
            A = np.zeros((N,N))         # 转移矩阵
            B = np.zeros((N, M+1))      # 状态释放观测值的概率矩阵
            PI = [0 for i in range(N)]  # 初始化的状态分布(暂时假定均匀分布)

            for i in range(N):
                for j in range(N):
                    A[i,j] = transition_prob[i][j]

            for i in range(N):
                for j in range(M+1):
                    if i < N-2:
                        B[i,j] = emission_prob[i][j]
                    else:
                        B[i,j] = 0.0
                        
            for i in range(N):
                PI[i] = state_prob[i]  
            
#             PI = [1/3, 1/3, 1/3, 0.0, 0.0]
#             PI = [11/36, 11/36, 11/36, 3/36, 0.0]
            path, max_pro = viterbi(O, Q, PI, A, B)
            
            
            # Join "BEGIN" and "END".
            output = []
            output.append(state_set['BEGIN'])
            output.extend(path)
            output.append(state_set['END'])
            output.append(max_pro)
            print(output)

def viterbiK(O, Q, PI, A, B, K):
    '''
    :param O: observations
    :param Q: states
    :param PI: state probability
    :param A: transition probability
    :param B: emission probability
    :param K: top-K
    :return path: the most possible state path
    :return prob: the largest probability  
    :IDEA: FOR EACH LOCAL STATE IN DP, WE COMPUTE TOP-K PATHS. WE NEED TO USE A PRIORITY_QUEUE TO STORE THE K PATHS
    '''
    
    # Special case: K=1
    if K == 1:
        return viterbi(O, Q, PI, A, B)
    
    # 初始化的过程
    # Step 0: Define three matrix -- delta, psi, rank.
    N = len(Q)
    T = len(O)    
    # For top-K, we have different definitions here!!!
    # delta[t,i,k] -- 在时间t，状态i条件下，第k个概率的值是多少
    delta = np.zeros((T,N,K), float)      # For each observation and state, top-k prob
    # psi[t,i,k] -- 在时间t，状态i条件下，第k个概率的值对应的状态值
    psi = np.zeros((T,N,K), int)          # Top-k most possible state at t    
        
    # Step 1: Initialize local states.
    for i in range(N):
        delta[0,i,0] = PI[i]*B[i,O[0]]   # 
        psi[0,i,0] = i
        
        for k in range(1,K):
            delta[0, i, k] = 0.0         # when t=1, init k probs for each state
            psi[0, i, k] = i             # when t=1, init k state for each state =>itself
       
    # Step 2: Continue DP to compute local top-k states in t = 1,3,...,T-1.
    for t in range(1, T):
        # Consider each state s2 (t) from previous state s1 (t-1).
        # 考虑时刻t以及状态s2的条件下
        for s2 in range(N):              # when t and s2
            # 定义一个vector存放的是概率+状态，并且按照概率有大到下
            prob_state = []              # define a priority_queue
            for s1 in range(N):          # state at t-1
                for k in range(K):       
                    prob = delta[t - 1, s1, k] * A[s1, s2] * B[s2, O[t]]
                    state = s1  
                    prob_state.append((prob, state))
            
            # Sort in descending order of prob and then state in ascending order.
            prob_state_sorted = sorted(prob_state, key=lambda x: x[0], reverse=True)

            # Update delta and psi value under t and s2.
            # 放回到状态s2对应的k个最大概率值，以及对应的状态
            for k in range(K):
                delta[t, s2, k] = prob_state_sorted[k][0] # when t and s2, top-k prob
                psi[t, s2, k] = prob_state_sorted[k][1]   # when t and s2, top-k state
        print(delta)
    
    # Step 3: Compute the top-K delta value at T (t=T-1), which is the probability of most possible state sequence.   
    prob_state = []                # Put all the last items on the stack.
    # Get all the topK from all the states.
    for s in range(N):
        for k in range(K):
            prob = delta[T - 1, s, k]
            # Store K <prob, state> pair for each state.
            prob_state.append((prob, s))
            
    # Sort by the probability and then state ID
    prob_state_sorted = sorted(prob_state, key=lambda x: x[0], reverse=True)
    
    # Step 4: Backtracking for k and t = T-1, T-2, ..., 1.
    path = np.zeros((K, T), int)         # initialize blank path
    path_prob = np.zeros(K, float)       # initialize max path probability
    
    for k in range(K):       
        max_prob = prob_state_sorted[k][0]        # max probability
        state = prob_state_sorted[k][1]           # corresponding state        

        path_prob[k] = max_prob
        path[k][-1] = state              # path is from tail to head

        # Backtrack each top-K path.
        for t in range(T-2, -1, -1):            
            new_state = psi[t+1][path[k][t+1]][k]  # path[k][t+1] is following state of new_state
            path[k][t] = new_state
        
    return path, path_prob
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    '''
    :param State_File: state file
    :param Symbol_File: symbol file
    :param Query_File: query file
    :param k: top-K
    :return output: return value
    '''
    
    # Generate state information.
    N, state_set, transition_prob, state_prob = read_state(State_File)
    
    # Generate symbol information.    
    M, symbol_set, emission_prob = read_symbol(Symbol_File, state_set)
    
    # Starting query.
    with open(Query_File, 'r') as file:
        while True:
            # Parse each line.
            line = file.readline()
            if not line:
                break
            query_seq = parse_query(line)      
            
            # Generate observations and initialized state probabiltiy.
            O = [M for i in range(len(query_seq))]
            for i in range(len(query_seq)):
                if query_seq[i] in symbol_set.keys():
                    O[i] = symbol_set[query_seq[i]]

            Q = range(N)
            
            # Convert dict into matrix -- A and B.
            A = np.zeros((N,N))
            B = np.zeros((N, M+1))
            PI = [0 for i in range(N)]

            for i in range(N):
                for j in range(N):
                    A[i,j] = transition_prob[i][j]

            for i in range(N):
                for j in range(M+1):
                    if i < N-2:
                        B[i,j] = emission_prob[i][j]
                    else:
                        B[i,j] = 0.0
                        
            for i in range(N):
                PI[i] = state_prob[i]          
            
            path, path_prob = viterbiK(O, Q, PI, A, B,k)

            for i in path:
                print(i)
                        
            # Join "BEGIN" and "END".
            # for k in range(k):
            #     output = []
            #     output.append(state_set['BEGIN'])
            #     output.extend(path[k])
            #     output.append(state_set['END'])
            #     output.append(path_prob[k])
            #     print(output)

if __name__ == "__main__":
    State_File ='./toy_example/State_File'
    Symbol_File='./toy_example/Symbol_File'
    Query_File ='./toy_example/Query_File'
    # viterbi_result = viterbi_algorithm(State_File, Symbol_File, Query_File)
    viterbi_result = top_k_viterbi(State_File, Symbol_File, Query_File, k=2)
