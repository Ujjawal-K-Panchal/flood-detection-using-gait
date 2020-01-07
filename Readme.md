---------------------------------------------
Flooding Level Classification by Gait Analysis of Smartphone Sensor Data

[+] Guide - Saad Yunus Sait.
[+] Researchers - Ujjawal K Panchal & Hardik Ajmani.
-----------------------------------------------

[+] Abstract :-
Urban flooding is a common problem across the world. In India, it leads to casualties every
year, and financial loss to the tune of tens of billions of rupees. The damage done due to flooding can be
mitigated if the locations deserving attention are known. This will enable an effective emergency response,
and provide enough information for the construction of appropriate storm water drains to mitigate the effect
of floods. In this work, a new technique to detect flooding level is introduced, which requires no additional
equipment, and consequent installation and maintenance costs. The gait characteristics in different flooding
levels have been captured by smartphone sensors, which are then used to classify flooding levels. In order to
accomplish this, smartphone sensor readings have been taken by 12 volunteers in pools of different depths,
and have been used to train machine learning models in a supervised manner. Support vector machines,
random forests and naïve bayes models have been attempted, of which, support vector machines perform
best with a classification accuracy of 99.45%. Further analysis of the most relevant features for classification
agrees with our intuition of gait characteristics in different depths.
-----------------------------------------------
[+] Repo Information:-

    1. RAW data from mobile sensors is available at data/raw/raw_data_reduced.csv  

    2. Pre - processing is done by code/preprocessing/transformation.py

    3. Pre - preprocessed data is then stored at data/transformed/preprocessed_data.csv
    
    4. Windowed data generation is done at code/window analysis/windowed_data_generation.py


    5. Windowed data is available at data/windowed/window_50_stride_25.csv
    
    6.  Models are available in code/models

    5.  Old contains the old folders and unused files.

    6.  Saad contains Saad sir's scripts.
-----------------------------------------------