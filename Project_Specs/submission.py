#!/usr/bin/env python3
# coding=UTF-8
'''
@Description: 
@Author: Peng LIU, ZhiHao LI
@LastEditors: Peng LIU
@Date: 2019-03-29 23:14:10
@LastEditTime: 2019-04-25 14:58:57
'''

# Import your files here...
import json
import re
import numpy as np

# deal with state_file
def read_StateFile(State_File):
    with open (State_File,'r') as file:
        N = int(file.readline())              # integer N , which is the number of states
        state_set = dict()                    # store the set of state
        transition_prob = dict()              # store transition probability  
        state_prob = dict()                   # store state initialising probability
        ID = 0                                # ID of states
        count = 0                             # number of transitions

        # Scan descriptive name of the states.
        while ID < N:
                state_name = file.readline().strip()
                state_set[state_name] = ID
                ID = ID + 1
                
        # Scan the transitions and weight.
        while True:
            line = file.readline()
            if not line:
                break
            items = line.split()

            state1 = int(items[0])      # The state before the transition
            state2 = int(items[1])      # The state after the transition
            weight = int(items[2])      # The weight of the transition

            transition_prob.setdefault(state1,{})
            transition_prob[state1][state2] = weight
            count = count + 1
        
        # Convert weight into probability.
        for keys,values in transition_prob.items():
            total = 0
            for value in values.values():
                total += value
            # Scan each state in state_set.
            for state in state_set.values():
                # Case 1: state is already existing
                if state in values.keys():
                    # A[i,j] = (n(i,j)+1)/(n(i)+N-1)
                    transition_prob[keys][state] = (transition_prob[keys][state]+1)/(total+N-1)
                # Case 2: state is not existing
                else:
                    if state == state_set['BEGIN']:
                        # For the BEGIN state, there is no transition to it, i.e., the probability is indeed 0.0.
                        transition_prob.setdefault(keys,{})[state] = 0.0
                    else:
                        transition_prob.setdefault(keys,{})[state] = 1/(total+N-1)
                        
    file.close()
    return transition_prob

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...


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
    viterbi_result = read_StateFile(State_File)
    print(viterbi_result)