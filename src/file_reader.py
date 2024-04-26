import string
import os

def read_action_seqs(filename):
    '''
    Read the input data and return list of action sequences.
    Each sequence is a list of action-argumentlist tuples.
    '''
    sequences = []
    filepath = os.getcwd() + '/data/' + filename
    file = open(filepath, 'r')
    for seq in file:
        actions = []
        arguments = []
        if seq and not seq.isspace() and len(seq)>1:
            sequence = seq.rstrip("\n\r").lstrip("\n\r").lower() 
            action_defs = sequence.split("),")

            for action_def in action_defs:
                action = action_def.split('(')[0].strip(")\n\r").strip()
                argument = action_def.split('(')[1].strip(")\n\r")
                actions.append(action.translate(str.maketrans('', '', string.punctuation)))
                argument_list = argument.split(',')
                argument_list = [x.strip() for x in argument_list]
                #argument_list.insert(0,'zero')
                arguments.append(argument_list)
                
            
            actarg_tuples = zip(actions,arguments)
            sequences.append(list(actarg_tuples))
    return sequences

def print_sequences(sequences):
    for seq in sequences:
        for index,action in enumerate(seq):
            print(str(index) + ": " + str(action))
        print("="*50)

# test
seqs = read_action_seqs('test_input.txt')
print_sequences(seqs)
