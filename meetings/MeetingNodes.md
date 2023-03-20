# Meeting notes - 2023-2-23
## Random geometric

- Pay attention to plane geometry of starting conditions, then use that intuit/test hypothesis about how the starting condition is chosen.

- Plot distribution of centrality measure

## B-A

- "Spread of plots from starting conditions"

## Generic comments

- Mean degree too high.

- Include degree distribution and also more facts e.g. max degree

- Implement data collapse on some of the curves shown in the plots.

- There was a comment about calculating farness however the closeness centrality is the reciprocal of the farness of a given node so who knows.

# Meeting notes - 2023-3-7

- Check that all historical data containing random geometric graphs was generated with uniform rather than Gaussian distribution (MHH note: I think this is the case)

- Check distances from seed node to other nodes (distribution of, mean, maximum, etc.)

- Probability scaling: Does a $p < 1 $ transmission probabiility imply some kind of neat time scaling (e.g. transformation to lower probability equivalent to $t \rightarrow \lambda t$)

- Usual comment about comparing paired dynamics.

- Plot the size of largest component at each time-step/edge-added. Probably best to plot sizes of largest
activated component and largest non-activated component (should seem some dynamics there to confirm hypothesis about large clumps being activated.)

- Plot/record "something" about the starting node of the BA graph e.g. compare intuition with reality when it come to age of starting node (wrt to preferential attachment and how the graph is constructed)


# Meeting notes - 2023-3-14

Charlie: 
- Point about matched comparisons again – plot the difference in saturation between RG and BA on the same plot for example.

- I have a note about circles and random geometric cases have inhomogeneity at the corners.

- Plot variation in how long reversible edges take to turn off, currently 10, what about N

- Plot variation in 50% time (denoted $T_{50}$): Reversible parameters, transmission proability and so on.




Max: 
- Wants to see scaling (not sure), and definitely a plot of data transformed into a straight line (e.g. inverse logistic).

- $\frac{\partial T_{50}}{\partial p}$? do physics blah look for ode dick james noises.

- Seed node statisics farness and so on.


Not this week: 
- There was a point about getting plots into a state fit for presentation to a human being.

DONE:
- Difference in path length distribution (seed )
