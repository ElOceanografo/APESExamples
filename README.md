# Demonstrations of an Automated Probabilistic Echo Solver for acoustic backscatter

This repository contains data and code to reproduce the examples, analyses, and figures
from "A  Bayesian inverse approach to identify and quantify organisms from fisheries acoustic data" by Samuel Urmy, Alex De Robertis, and Christopher Bassett (2023), *ICES Journal of Marine Science*,  https://doi.org/10.1093/icesjms/fsad102.

<<<<<<< HEAD
To run the example problems, you will need to [install Julia](https://julialang.org/downloads/) (v1.8 or higher) and run it in this directory. This project includes a self-contained environment for reproducibility. To install all packages at the version used for the publication, type `]` at the Julia command line to enter package-manager mode. The prompt should change from `julia>` to `(@v1.9) pkg>`. From here, run the following commands to activate the APESExamples environment and download all required packages.
=======
To run the example problems, you will need to [install Julia](https://julialang.org/downloads/) (v1.9 or higher) and run it in this directory. This project includes a self-contained environment for reproducibility. To install all packages at the version used for the publication, type `]` at the Julia command line to enter package-manager mode. The prompt should change from `julia>` to `(@v1.9) pkg>`. From here, run the following commands to activate the APESExamples environment and download all required packages.
>>>>>>> 55b8b659d42f3027007983a8ee16f67e080b0711

```
(@v1.9) pkg> activate .
  Activating project at <wherever you saved the repository>

(APESExamples) pkg> instantiate
```

Hit "backspace" to exit the package manager and return to the `julia>` prompt.

The analyses in this package all rely on [ProbabilisticEchoInversion.jl](https://github.com/ElOceanografo/ProbabilisticEchoInversion.jl), which contains the actual implementation of APES. Refer to that package's documentation for more detail on how to implement your own models.

Each subdirectory here contains a script to reproduce one of the analyses from the paper. You can open these in your favorite IDE ([VSCode](https://code.visualstudio.com/) with the [Julia extension](https://www.julia-vscode.org/) is currently the best supported) and step through them interactively, or run them as scripts by `include`-ing them, for example:

```julia
include("fish_krill_simulation/fish_krill.jl")
```

<<<<<<< HEAD
Note that the Aleutian narrowband and Barnabas broadband examples will attempt to use all available cores on your machine and will take an hour or more to run, depending on your
computer's speed and number of processors.
=======
Note that the Aleutian narrowband and Barnabas broadband examples will attempt to use all available cores on your machine and will take on the order of an 1-2 hours to run, depending on you computer's speed and number of processors.
>>>>>>> 55b8b659d42f3027007983a8ee16f67e080b0711
