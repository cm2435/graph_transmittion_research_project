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

