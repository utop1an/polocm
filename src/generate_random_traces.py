import os
import importlib.util
import json
from generate.pddl import Generator
from planner.random_planner import RandomPlanner
from utils import TraceSearchTimeOut
import signal

def handler(signum, frame):
    raise TimeoutError("Execution time exceeded")

# Get the list of directories in the 'data/classical' folder
dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'classical'))
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'json_traces'))

directories = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
directories.sort()

with open(os.path.join(out_dir, "random.txt"), 'r') as file:
    lines = file.readlines()

existing = ["&&".join(x.split("&&")[0:-2]) for x in lines]

with open (os.path.join(out_dir, "random.txt"), 'a') as file:
    # For each directory
    for directory in directories:
        # Construct the path to the api.py file
        api_file_path = os.path.join(dir, directory, 'api.py')

        # Check if the api.py file exists
        if os.path.exists(api_file_path):
            # Load the api.py module
            spec = importlib.util.spec_from_file_location('api', api_file_path)
            api = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api)

            # Access the 'domains' variable
            domains = api.domains
            # Now you can use the 'domains' variable
         
            for domain in domains:
               
                for problem in domain['problems']:
                    
                    domain_file, problem_file = problem
                    domain_name = domain_file.split('.')[0]
                    problem_name = problem_file.split('.')[0]
                    domain_file_path = os.path.join(dir, domain_file)
                    problem_file_path = os.path.join(dir, problem_file)
                    print(f"Generating traces for: \n\tDomain: {domain_name}, Problem: {problem_name}")
                    
                    # g = Generator(domain_file_path, problem_file_path)
                    # try: 
                    #     plan = g.generate_plan()
                    # except Exception as e:
                    #     print(f"Error generating plan: {e}")
                    #     continue
                    # print(plan)
                    # plan_trace = g.generate_single_trace_from_plan(plan)
                    if (f"{domain['name']}&&{domain_name}&&{problem_name}" in existing):
                        if (domain_name.split("/")[-1] == 'domain.pddl'):
                            print(f"Skipping {domain['name']}")
                            break
                        print(f"Skipping {domain_name} {problem_name}")
                        continue
                    if (domain_name.split("/")[-1] == 'domain'):
                        settings = [(50,1)]

                    for plan_len, num_traces in settings:
                        signal.signal(signal.SIGALRM, handler)
                        signal.alarm(40)
                        try: 
                            rand_planner = RandomPlanner(domain_file_path, problem_file_path, plan_len=plan_len, num_traces=num_traces)
                            traces = rand_planner.generate_traces()
                            for i, trace in enumerate(traces):
                                t = [op.name for op in trace]
                                output = f"{domain['name']}&&{domain_name}&&{problem_name}&&{plan_len}&&{','.join(t)}\n"
                                file.write(output)
                        except TimeoutError as te:
                            print(f"Timeout generating traces: {te}")
                            continue
                        except TraceSearchTimeOut as tst:
                            print(f"Timeout trace search: {tst}")
                            continue
                        except Exception as e:
                            print(f"Error generating traces: {e}")
                            continue
                        finally:
                            signal.alarm(0)

