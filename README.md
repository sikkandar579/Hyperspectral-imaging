# Hyperspectral-imaging
Detection of Pesticides Residues in Lady's finger using hyperspectral imaging
Objective:
-> The objective of this project was to classify lady's finger (okra) into four distinct categories based on pesticide concentration: pure (without insecticide), low concentration (1 mg/L), medium concentration (3 mg/L), and high concentration (5 mg/L). To achieve this, we prepared samples by applying different concentrations of the insecticide M Power to lady's finger.

-> The next step involved wavelength selection, where we identified Regions of Interest (ROI) from hyperspectral images. By comparing the mean spectral curves for samples with and without pesticides, we identified optimal wavelengths. The chosen wavelength ranges were 388.85 to 660 nm (Visible region) and 720 to 1000 nm (NIR region).

-> For feature extraction, we obtained features based on the minimum and maximum reflectance values within the selected wavelength regions. We extracted 100 features for classification, with 80 used for training and 20 for testing. We implemented five classifiers: Ensemble, Support Vector Machine (SVM), K-Nearest Neighbour (KNN), Naive Bayes, and Logistic Regression. Among these, Logistic Regression yielded the highest accuracy.

-> We faced a challenge as the mean spectral curves for low vs. pure and medium vs. high pesticide concentrations overlapped in the NIR region, complicating four-class classification. To address this, we employed deep learning models, specifically 3D Convolutional Neural Networks (3D CNN). We tested two architectures: Squeeze-Excitation Residual Spectral Network (SERSN) and a standard Residual Network without SE(Squeeze Excitation). We achieved 97% accuracy with the SE block and 89% without SE(Squeeze Excitation).

-> This project successfully demonstrated the ability to distinguish between pesticide concentrations using hyperspectral imaging and advanced deep learning techniques. The SERSN model proved particularly effective in improving classification accuracy.
