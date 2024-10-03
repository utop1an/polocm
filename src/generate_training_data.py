import os
import pandas as pd
import random
import json
import argparse
import uuid

SEED = 42
REPEAT = 1
TRACELENGTH = [10,20,50,100]
NUMBEROFTRACES =[1,5,25,50,100]
COUNTER = 0

def write_to_file(output_data, file_path):
    try:
        with open(file_path, 'w', buffering=1) as file:  # Line buffered
            json.dump(output_data, file)
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

def sample_combined(df, number_of_traces):
    types = df['type'].unique()
    p1 = df[df['type'] == types[0]]
    p2 = df[df['type'] == types[1]]

    p1_traces = p1.sample(n=number_of_traces//2, random_state=SEED)
    p2_traces = p2.sample(n=number_of_traces-len(p1_traces), random_state=SEED)

    return pd.concat([p1_traces, p2_traces])

def generate_trace(domain, df, number_of_traces,combined, trace_length=None, diff=""):
    global COUNTER
    if combined == "combined":
        rows = sample_combined(df, number_of_traces)
    elif combined == "plan":
        rows = df[df['type'] == 'plan'].sample(n=number_of_traces, random_state=SEED)
    elif combined == "random":
        rows = df[df['type'] == 'rand'].sample(n=number_of_traces, random_state=SEED)
    output = []
    for i in range(REPEAT):
        traces = []
        total_length = 0
        number_of_objects = 0
        for r, row in rows.iterrows():
            plain_trace = row['trace'].split(',')
            if trace_length is not None:
                if (len(plain_trace) <= trace_length):
                    rand_trace = plain_trace
                else:
                    rand_start = random.randint(0, len(plain_trace) - trace_length)
                    rand_trace = plain_trace[rand_start:rand_start + trace_length]
            else:
                rand_trace = plain_trace
            trace = []
            total_length += len(rand_trace)
            number_of_objects += int(row['number_of_objects'])
            for plain_op in rand_trace:
                op = plain_op.strip('()').split(' ')
                trace.append({'action': op[0], 'objs': op[1:]})
            traces.append(trace)
        output_obj = {
            'id': COUNTER,
            'domain': domain,
            'index': i,
            'difficulty': diff,
            'total_length': total_length,
            'traces': traces,
            'number_of_objects': int(number_of_objects/len(traces))
        }
        output.append(output_obj)
        COUNTER += 1
    return output
        

def main(args):
    global SEED, REPEAT
    input_filepath = args.i
    output_dir = args.o
    seed = args.s
    repeat = args.r
    isDiff = args.d
    combined = args.c
    SEED = seed
    REPEAT = repeat
    if SEED is not None:
        random.seed(SEED)
    
    if combined not in ["combined", "plan", "random"]:
        print("Invalid combination type. Choose from combined, plan, random")
        return

    if not os.path.exists(input_filepath):
        print(f"Input file {input_filepath} does not exist")
        return
    

    headers = ['domain', 'type', 'problem_name', "difficulty", "number_of_objects", 'plan_len', 'trace']
    input_data = pd.DataFrame(columns=headers)
    with open(input_filepath, 'r') as file:
        for line in file:
            raw = line.strip().split('&&')

            if (raw[-1]!='Error' and raw[-1]!='TimeoutError' and raw[-1]!='TraceSearchTimeOut'):
                input_data.loc[len(input_data)] = raw
    
    output_data = []
    domains = input_data['domain'].unique()
    difficulty = input_data['difficulty'].unique()
    for domain in domains:
        if isDiff:
            for diff in difficulty:
                df = input_data[(input_data['domain'] == domain) & (input_data['difficulty']==diff)]
                if (len(df)==0):
                    continue
                for length in TRACELENGTH:
                    for num in NUMBEROFTRACES:
                        if num > len(df):
                            break
                        output = generate_trace(domain, df, num, combined,trace_length=length, diff=diff)
                        output_data= output_data + output
        else:
            df = input_data[input_data['domain'] == domain]
            for length in TRACELENGTH:
                for num in NUMBEROFTRACES:
                    if num > len(df):
                        break
                    output = generate_trace(domain, df, num, combined,trace_length=length)
                    output_data= output_data + output
    
    output_filename = 'traces'
    if (isDiff):
        output_filename += '_diff'
    output_filename += f'_{combined}'
    output_filename += f'_r{REPEAT}.json'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_to_file(output_data, os.path.join(output_dir, output_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create json traces from plain traces')
    parser.add_argument('--i', type=str, help='Input plain trace file path')
    parser.add_argument('--o', type=str, help='Output directory')
    parser.add_argument('--s', type=int, default=42, help='Seed for random generation')
    parser.add_argument('--r', type=int, default=1, help='Number of times to repeat the generation')
    parser.add_argument('--d', type=bool, default=False, help='Generate traces for different difficulty levels')
    parser.add_argument('--c', type=str, default="combined", help='Generate traces combining plans and random walks or only pans or only random walks')
    args = parser.parse_args()

    main(args)