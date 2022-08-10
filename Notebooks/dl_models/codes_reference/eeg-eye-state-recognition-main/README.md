# eeg-eye-state-recognition
Using machine learning models to predict eye state of the subject based on preprocessed EEG signal. Includes preprocessing the data, feature extraction and selection, dimensionality reduction and visualization.

## Data
Data is from private database of 72 EEG recordings with duration of 40 minutes. Data was described by techinician and arranged into eye state intervals.

## Used tools
- numpy, pandas
- pywt, mne
- scikit-learn

## Used methods
- Wavelet transform (discrete)
- Principal Component Analysis
- Fischer Discriminant Ratio
- Manual selection of channels based on topographic power density maps with the use of DWT
- k-NN, XGBoost, Multilayer Perceptron, Support Vector Machine

## Results
Achived best accuracy of 88,2% with manual selection using MLP and KNN classificatiors. It's higher than previous work that I put below considering same classification problem (6-7 in Literature).


## Literature 
1. S. Sanei and J. A. Chambers, EEG Signal Processing. Wiley, 2013.
2. R. J. Barry, F. M. D. Blasio, J. S. Fogarty, and A. R. Clarke, “EEG differences between eyes-closed and eyes-open resting conditions,” Clinical Neurophysiology, vol. 131, no. 1, 2007.
3. Ling, Li, Lei, Xiao, Long, and Chen, “Differences of EEG between eyes-open and eyesclosed states based on autoregressive method,” 2009.
4. H. U. Amin, W. Mumtaz, A. R. Subhani, M. N. M. Saad, and A. S. Malik, “Classification of EEG signals based on pattern recognition approach,” Frontiers in Computational
Neuroscience, vol. 11, p. 103, 2017.
5. K. Sabanci and M. Koklu, “The classification of eye state by using kNN and MLP classification models according to the EEG signals,” International Journal of Intelligent Systems and Applications in Engineering, vol. 3, p. 127, 2015
6. U. N. Wisesty, H. Priabdi, R. Rismala, and M. D. Sulistiyo, “Eye state prediction based on
EEG signal Data Neural Network and Evolutionary Algorithm Optimization” 
7. T. Wang, S.-U. Guan, K. L. Man, and T. O. Ting, “EEG eye state identification using
Incremental Attribute Learning with time-series classification” 
