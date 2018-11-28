# LinearPredictogram


LinearPredictogram explores the Gray dataset using a linear regression trying to predict a signal from another from different brain regions and from different frequency band. It also transforms the raw data from MATLAB to NumPy-ready and applies the usual pre-processing steps as well as dealing with the analysis and visualization of the results


Its filtered the signal between :
7 and 12 Hz. 
It select time between :
-800ms before sample on to +600ms after sample off
-600 before match on to +2000ms after match on

It cuts signal into windows of 200 ms with a step of 100 ms.

A linear regression is fitted on a tranning base (80% of the dataset) and then test on a testing base (20% of the dataset).



