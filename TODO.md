1. Impliment adjacency matrix change so that each node per step can only have one connection (communities no larger than 2)

2. Impliment averaging of all the simulation runs, return list of all the infection arrays, and then plot as a function of time 
- vairy params like size of matrix, number agents ect.

3. Impliment graph viz stuff 

4. Fit equasions / research theory for the erdos-something graph structure. 

5. Barabasi graph look at  




24.11.2022

1. Look at the 'causal' case of edges turning on and off for the edge by edge saturation behaviour of networks
2. Look at the 'reachability' matrix of all nodes in the graph as a function of the timesteps for this 

Extentions: 
1. Look at:
    variable length timesteps
    nodes being able to turn off 
    various edge weights 
    different distributions of static graph structures 



# TODO FOR THIS WEEK 
# Take the reachability and impliment it to find the average of the matrix for each timestep and plot
# Charlie is dumb and wrote the core logic as dicts when the reachility matrix function takes in np arrays, core logic needs refactor to be done in numpy 


# 14.02.2023
# Areas for refinement
# Compare barabasi to geometric graph
# Look at that is definitely the structure we think it is 

Look at the grid of two 'vanilla' structures and irreversable and reversable

Look at transforms of saturation curve to see if we can ensure that is the right behaviour 

Things to control for: 
    Mean degree 
    Choice of starting node 
    Num agents 

Things to vairy: 
    Mean path length (graph structure)
    Community existance (graph structure)
    