# fuzzy-highway-bottleneck-python
Python-based code for estimation of highway bottleneck probability using speed transition matrices. 
**Full paper with detailed explanation will be available soon!** 

## Info
Bottlenecks on the highways are caused by two types of congestion: 1) recurrent ones, which happens in the rush-hours due to a large number of commuters, and 2) non-recurrent ones that happen after some unexpected event like a traffic accident or sudden breaks. Bottlenecks are formally defined as a decrease in the capacity of the highway due to a congestion event. 
This code implements the fuzzy method for estimation of the highway bottleneck probability occurrence using a novel traffic data model called Speed Transition Matrix (STM). The method is based on a Python package for fuzzy inference systems [simpful](https://github.com/aresio/simpful) and implemented class for STM computation. 
The data is proveded in the file `TrafData.pkl`. It contains the routes of the vehicles on a highway simulated in the [SUMO simulator](https://www.eclipse.org/sumo/). 

## Requirements
1. Install Python (3.8 recommended) [Download link](https://www.python.org/downloads/).
2. Install required packages from `requirenments.txt` using [virtual environment](https://docs.python.org/3/tutorial/venv.html).

## More info
You can expect more info and the scientific paper and Medium article by September 2021.

## Contact and connect
[Leo Tisljaric](https://www.linkedin.com/in/leo-ti%C5%A1ljari%C4%87-28a56b123/)

