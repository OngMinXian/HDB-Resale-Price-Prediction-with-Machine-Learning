# HDB Resale Price Prediction with Machine Learning

### About the project
---
The project aims to predict HDB resale price using the TensorFlow framework. The Pandas framework was used to hold the data and Matplotlib was used to plot the graps. This project was started after completion of the Machine Learning Specialization course by Andrew Ng on Coursera and aims to put the skills learnt into a dataset relatable to me and without any guidance.

### Dataset
---
The dataset was provided by Data.gov.sg and can be found [here](https://data.gov.sg/dataset/resale-flat-prices). While data of resale prices can be found from 1990 to 2023, only the 2017 - 2023 dataset will be used as resale prices before that might not be as indicative of resale prices now and in the future. 

The features include date of purchase, town, flat type, block, street name, storey range, floor area, flat model, lease commence date, remaining lease and our label is resale price. The plots of certain features against resale price will help decide which features to keep and numerised.

<img src="https://github.com/OngMinXian/HDB-Resale-Price-Prediction-with-Machine-Learning/blob/main/months_past_2017_Jan.png"> <br />
The date data is in the form of year and month and will be converted to number of months past January 2017 to keep a single numerical value that increases. The number of months past January 2017 shows an increasing trend with resale price except the priod of 2019 to 2020 which is due to COVID19.

<img src="https://github.com/OngMinXian/HDB-Resale-Price-Prediction-with-Machine-Learning/blob/main/town.png"> <br />
Boxplot of each town is plotted and there shows a correlation between town and resale price. Town data is processed with one hot encoding. Interesting patterns can be seen such as the central areas and the areas near bukit timah having a higher average and interquatile range. 

<img src="https://github.com/OngMinXian/HDB-Resale-Price-Prediction-with-Machine-Learning/blob/main/flat_type.png" width="800" height="500"> <br />
Boxplot of each flat type is plotted and there shows a correlation between flat type and resale price. Flat type is processed with one hot encoding as well. An expected pattern can be observed that as there are more rooms, the price of the resale would be higher.

<img src="https://github.com/OngMinXian/HDB-Resale-Price-Prediction-with-Machine-Learning/blob/main/storey.png"> <br />
To process the story range, a random number is generated from the range to be used as the storey. Resale price is grouped by storey and the average is plotted out. There is an an observed trend where the higher the storey, the higher the resale price. Reasons that could explain the trend are nicer views, quieter environment and better air quality found on the higher floors.

<img src="https://github.com/OngMinXian/HDB-Resale-Price-Prediction-with-Machine-Learning/blob/main/floor_area_sqm.png"> <br />
A scatterplot of floor area against resale price shows an increasing trend. This is to be expected as people generally prefers a bigger apartment.

<img src="https://github.com/OngMinXian/HDB-Resale-Price-Prediction-with-Machine-Learning/blob/main/flat_model.png"> <br />
Boxplot of each flat model is plotted and there shows a correlation between flat model and resale price.

<img src="https://github.com/OngMinXian/HDB-Resale-Price-Prediction-with-Machine-Learning/blob/main/remaining_lease_months.png"> <br />
Plot of remaining lease months shows an increasing trend. This is as expected as the higher the remaining lease months, the higher the value as once the lease is over, HDB reclaims the flat.

Block and street name were left out as features due to town being a good representative of them. Including those 2 features might result in overfitting. Lease commence date can be considered the same information as remaining lease months. In future projects, outliers can be removed from the dataset before training the model. The data is the normalized and then split into training and test data with 9:1 ratio respectively. During training, 1/8 of the training data is used as cross validation data.

### Training the model
---
##### Linear regression model
The data was first fitted to a simple linear regression model. The cross validation results can be shown below and the cost was 0.045.
<img src="https://github.com/OngMinXian/HDB-Resale-Price-Prediction-with-Machine-Learning/blob/main/linear_model_loss.png"> <br />

##### Neural network model
---
The data was then fitted to a neural network model that consisted of 4 layers which 32, 16, 8 and 1 unit respectively. 'relu' activation and L2 regularizers were used. The model was compiled with Adam optimizer and loss function is mean absolute error. The cost after fitting was 0.033.
<img src="https://github.com/OngMinXian/HDB-Resale-Price-Prediction-with-Machine-Learning/blob/main/neural_network_model_loss.png"> <br />

##### Refining the neural network model
---
In order to refine the neural network, the learning rate for the Adam optimizer and the lambda value for the regularizer was varied. By fitting a neural model with the given parameters, we can find the best learning rate and lambda value to obtain the lowest cost. This turned out to be 0.022. However, there are some fluctuations with the cost graph and this could suggest that there has been overfitting, However, due to the significant decrease in cost and acceptable level of fluctuations, this model was kept.
<img src="https://github.com/OngMinXian/HDB-Resale-Price-Prediction-with-Machine-Learning/blob/main/final_neural_network_model_loss.png"> <br />

Models were saved in h5 files which can be found in the repository.

### Conclusion
---
Finally, the refined neural network model was tested against the test data. The final cost was 0.023 which is close to cross validation and training data cost. This suggests that there is low bias and variance and thus, no underfitting or overfitting. 

10 randomly chosen resale flats are shown below showing the difference in predicted and actual.
| Predicted | Actual |
|-----------|--------|
| 295602    | 262000 |
| 302733    | 288500 |
| 338846    | 321000 |
| 354275    | 335000 |
| 351532    | 373000 |
| 478019    | 518000 |
| 686999    | 688000 |
| 802243    | 888000 |
| 317393    | 300000 |
| 319095    | 310000 |
