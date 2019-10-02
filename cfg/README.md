# Configuration Files

Put configuration files for experiments and environments. Put them in `.yaml`
format. 

For the sake of keeping experiments reproducible and to avoid extensive tuning,
use one `.yaml` file per example script we want to run, where the scripts may
require different parameters.

We don't follow OpenAI gym conventions exactly. Normal gym environments assume
exact stability in the environment definitions. But we want to pass in our
`.yaml` configurations.

Most hyperparameters are straightforward and hopefully documenting one of the
configuration files will be sufficient.
