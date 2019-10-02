# Analysis

All plotting scripts go here, scripts for checking data statistics and
processing it, etc.

For starting state and demonstration data statistics, generated from
`examples/analytic.py`, use these two scripts for analysis:

- `combine_demo_data.py` -- combine demonstrator data, we run in separate python
  scripts since each call to `python examples/analytic.py` just requires one
  CPU. Also conveniently helps us record our data management, i.e., which files
  correspond to which settings, and which ones were used to combine to form a
  larger dataset.

- `check_demo_data.py` -- investigate performance of the analytic policies by
  loading data from `combine_demo_data.py`. After August 7, we have starting
  state information stored in the environment, which lets us see starting state
  statistics.
