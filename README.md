# fuzzy-highway-bottleneck-python :articulated_lorry:
Python-based code for estimation of highway bottleneck probability using speed transition matrices.  
**Full paper with detailed explanation will be available soon!** 

## Info
Bottlenecks on the highways are caused by two types of congestion: 1) recurrent ones, which happens in the rush-hours due to a large number of commuters, and 2) non-recurrent ones that happen after some unexpected event like a traffic accident or sudden breaks. Bottlenecks are formally defined as a decrease in the capacity of the highway due to a congestion event.  
This code implements the fuzzy method for estimation of the highway bottleneck probability occurrence using a novel traffic data model called [Speed Transition Matrix (STM)](https://medium.com/analytics-vidhya/speed-transition-matrix-novel-road-traffic-data-modeling-technique-d37bd82398d1). The method is based on a Python package for fuzzy inference systems [simpful](https://github.com/aresio/simpful) and implemented class for STM computation.  
The data is proveded in the file `TrafData.pkl`. It contains the routes of the vehicles on a highway simulated in the [SUMO simulator](https://www.eclipse.org/sumo/). 

## Requirements
1. Install Python (3.8 recommended) [Download link](https://www.python.org/downloads/).
2. Install required packages from `requirenments.txt` using [virtual environment](https://docs.python.org/3/tutorial/venv.html).

## More info
Medium article [https://towardsdatascience.com/using-fuzzy-logic-for-road-traffic-congestion-index-estimation-b649f8ddede1](https://towardsdatascience.com/using-fuzzy-logic-for-road-traffic-congestion-index-estimation-b649f8ddede1).

## Cite as
Text:  
Tišljarić, Leo, Filip Vrbanić, Edouard Ivanjko, and Tonči Carić. 2022. "Motorway Bottleneck Probability Estimation in Connected Vehicles Environment Using Speed Transition Matrices" Sensors 22, no. 7: 2807. https://doi.org/10.3390/s22072807

.bib:  
@Article{s22072807,
AUTHOR = {Tišljarić, Leo and Vrbanić, Filip and Ivanjko, Edouard and Carić, Tonči},
TITLE = {Motorway Bottleneck Probability Estimation in Connected Vehicles Environment Using Speed Transition Matrices},
JOURNAL = {Sensors},
VOLUME = {22},
YEAR = {2022},
NUMBER = {7},
ARTICLE-NUMBER = {2807},
URL = {https://www.mdpi.com/1424-8220/22/7/2807},
PubMedID = {35408421},
ISSN = {1424-8220},
DOI = {10.3390/s22072807}
}

## Contact and connect
[Leo Tisljaric](https://www.linkedin.com/in/leo-tisljaric-28a56b123/)

