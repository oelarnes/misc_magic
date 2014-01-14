This folder contains miscellaneous projects related to analysis of Magic: the Gathering
tournament outcomes.

84winrates.py analyses the effect of single-elimination tournaments on the distribution
of Elo ratings across rounds

elosimulator.py simulates an implementation of Elo and Glicko rating schemes over the
course of many matches in a fixed population. In particular, we wish to determine the
impact of cutoffs for high-level play based on ratings thresholds. The simulations
demonstrate that such cutoffs cause ratings points to drift artificially across the 
ratings barrier.

resultscompiler.py takes win-loss records and decklists hosted online in a standard
format, and returns a csv file with win-loss records for each card appearing in a 
decklist, with the purpose of allowing statistical analysis on the strength of 
individual cards.