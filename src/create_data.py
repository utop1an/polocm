import os
import pandas as pd
import random
import json
import argparse

SEED = 42
REPEAT = 1
TRACELENGTH = [10,20,50,100]
NUMBEROFTRACES =[1,5,25,50,100]
DIFFICULTY = ['easy', 'medium', 'hard']

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
    if combined:
        rows = sample_combined(df, number_of_traces)
    else:
        rows = df.sample(n=number_of_traces, random_state=SEED)
    output = []
    for i in range(REPEAT):
        traces = []
        total_length = 0
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
            for plain_op in rand_trace:
                op = plain_op.strip('()').split(' ')
                trace.append({'action': op[0], 'objs': op[1:]})
            traces.append(trace)
        output_obj = {
            'domain': domain,
            'index': i,
            'difficulty': diff,
            'total_length': total_length,
            'traces': traces
        }
        output.append(output_obj)
    return output
        

def main(args):
    global SEED, REPEAT
    input_filename = args.i
    seed = args.s
    repeat = args.r
    isDiff = args.d
    combined = args.c
    SEED = seed
    REPEAT = repeat
    if SEED is not None:
        random.seed(SEED)
    
    input_filename = input_filename.split("+")
    if combined:
        if len(input_filename) != 2:
            print("Please provide two input file names [plan+random] for combined traces")
            return
    else:
        if len(input_filename) != 1:
            print("Please provide only one input file name")
            return

    first_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data','plain_traces', f'{input_filename[0]}.txt'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'json_traces'))

    headers = ['domain', 'difficulty', 'problem_name', 'plan_len', 'trace', 'type']
    input_data = pd.DataFrame(columns=headers)
    with open(first_file, 'r') as file:
        for line in file:
            raw = line.strip().split('&&')
            raw.append(input_filename[0])
            if (raw[-1]!='Error' and raw[-1]!='TimeoutError' and raw[-1]!='TraceSearchTimeOut'):
                input_data.loc[len(input_data)] = raw
    
    if combined:
        second_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data','plain_traces', f'{input_filename[1]}.txt'))
        with open(second_file, 'r') as file:
            for line in file:
                raw = line.strip().split('&&')
                raw.append(input_filename[1])
                if (raw[-1]!='Error' and raw[-1]!='TimeoutError' and raw[-1]!='TraceSearchTimeOut'):
                    input_data.loc[len(input_data)] = raw
    
    output_data = []
    domains = input_data['domain'].unique()
    for domain in domains:
        if isDiff:
            for diff in DIFFICULTY:
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
    
    output_filename = "+".join(input_filename)
    if (isDiff):
        output_filename += '_diff'
    if combined:
        output_filename += '_combined'
    output_filename += f'_s{SEED}_r{REPEAT}.json'
    write_to_file(output_data, os.path.join(out_dir, output_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create json traces from plain traces')
    parser.add_argument('--i', type=str, help='Input plain trace file name without ending.If combined traces provide two file names separated by +')
    parser.add_argument('--s', type=int, default=42, help='Seed for random generation')
    parser.add_argument('--r', type=int, default=1, help='Number of times to repeat the generation')
    parser.add_argument('--d', type=bool, default=False, help='Generate traces for different difficulty levels')
    parser.add_argument('--c', type=bool, default=False, help='Generate traces combining plan and random walks')
    args = parser.parse_args()

    main(args)