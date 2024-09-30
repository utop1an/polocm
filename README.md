# xLOCM
NL to PDDL, based on LOCM

## ipc benchmarks
ADL, disjunction preconditions, equality, conditional effects... removed

### random walks
plan len of 50

# problems:
locm only checks inaps, but we should do inaps and outaps
    - check both

locm cannot handle complex :types and :constants
    - ignore?
    - find a way to enable hierarchy of sorts? Impossible for now
    - input the ground truth types, compare 

some domains are poorly defined, zenotravel, random generated traces might be strange, fly(airplane, city1, city1), invalid obj_trace
    - give an example of how it impacts on the learning algorithm

not a problem...random sampled traces might not cover all the actions...and (all) the transitions
    
comparing the executability with the locm2 result? using different traces as the trainning ones


