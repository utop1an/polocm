import itertools
import os
from tabulate import tabulate
from IPython.display import display, Markdown

def empty_directory(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

# utils
def findsubsets(S,m):
    return set(itertools.combinations(S, m))

def print_table(matrix):
    display(tabulate(matrix, headers='keys', tablefmt='html'))
    
def printmd(string):
    display(Markdown(string))