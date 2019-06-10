# Using Nearby Businesses to Predict Toronto Apartment Rental Prices 
### by Bailey Duncan

![Picture of Monopoly Houses Resting on a Pound Sterling Coin](./images/monopoly.jpg)

## Table of Contents
1. [Introduction](#1---introduction)
2. [Data Description](#2---data-description)
3. [Methodology](#3---methodology)
4. [Results](#4---results)
5. [Discussion](#5---discussion)
6. [Conclusion](#6---conclusion)
7. [References](#7---references)

## 1 - Introduction
Can we predict a Toronto apartment's rental price by knowing the businesses around it? This capstone is an exploration to see what effect businesses have on nearby apartment prices, and to try and determine the trends that lead to more expensive or desirable living space. If we know that certain businesses impact the price of an apartment when they open nearby, we can locate businesses that are more valuable to people. It can help land developers and business owners better forecast how new businesses can have an impact on rent prices of neighborhoods.


[To Top](#table-of-contents)

## 2 - Data Description
The data used will be a combination of datasets, and data from API calls. The primary dataset is compromised of Toronto apartment rental prices for 2018. The dataset contains the price, address, latitude, longitude, and rooms of 1124 apartments for rent in the Toronto area. This dataset is publicly available on Kaggle<sup>1</sup>. The Latitude, Longitude values in the dataset will be used with the Foursquare API to find the businesses nearby each apartment.

[To Top](#table-of-contents)

## 3 - Methodology 
Methodology section which represents the main component of the report where you discuss and describe any exploratory data analysis that you did, any inferential statistical testing that you performed, and what machine learnings were used and why.

### 3.1 - Data Cleansing
Several steps were taken to prepare the kaggle dataset for the project. The apartment prices were initially string values prefixed by a dollar sign, so they were converted to float values. The address feature was not interesting so I decided to extract the first three letters of the apartment postal code using a regex, and drop the rest of the address information. 

![Figure 1. Image of the uncleaned Kaggle dataset in a pandas dataframe](./images/initial_dataset.PNG)

*Figure 1. Image of the uncleaned Kaggle dataset in a pandas dataframe*

That way I could easily group apartments by the postal code area they reside in (it could also be done with the latitude/longitude, but not without a lot of effort setting up neighborhood boundaries). Next the apartments were mapped using their latitude/longitude so I could identify outliers. I found and removed several stray apartments in Montreal, Calgary, Winnipeg and other Ontario cities that should not have been in the Toronto dataset. I had a dataset listing Toronto neighborhoods for each postal code from a previous project<sup>2</sup>, so I used that to label the neighborhood and borough of each apartment in the dataset. 

![Figure 2. Stray Apartments in the Dataset](./images/toronto_apt_capstone_zoom4.PNG)

*Figure 2. Stray Apartments in the Dataset*

From this point I looked at the latitude/longitude columns and thought it looked like a less useful feature because of its scale, difficulty to work with, and lack of meaning to us without a map alongside it. I decided to derive a new feature from them called "Distance to Downtown Toronto" by calculating the euclidean distance from City Hall. Then I normalized it with a Min/Max Scaler to shift the values to all be between zero and one. That way all of our features are positive, and most of our features are in the zero-to-one range for any models that are sensitive to scaling. 

![Figure 3. Toronto Venue Data](./images/venue_information.PNG)

*Figure 3. Toronto Venue Data*

Next I used FourSquare to get the **venues** (FourSquare's term for businesses, amenities, parks, etc) within one kilometer of each apartment. I converted these venues into a one-hot encoded matrix, then aggregated all the rows related to one apartment, and merged it with the prepared kaggle dataframe. 

![Figure 4. Apartments data merged with the Venue Matrix](./images/apt_and_venue_df.PNG)

*Figure 4. Apartments data merged with the Venue Matrix*

### 3.2 - Exploratory Analysis

The bulk of the exploratory analysis was focused on the kaggle dataset. 

![Figure 5. Histogram of Toronto Apartment Prices](./images/apartment%20price%20histogram.PNG)

*Figure 5. Histogram of Toronto Apartment Prices*

Looking at the apartment prices distribution, we see it forms a nice bell curve centered around a $2500 rental price, ranging from around $500 to $5000. It is skewed right, which makes sense as the apartment prices have a hard limit before $0, but can be any value on the right side of the curve (ie. they extend from 0 to **∞**). This leads to a long trailing tail on the right side of our distribution and a short left side of the distribution.

![Figure 6.1 Histogram of Toronto Apartment Proximity to Downtown](./images/downtown_proximity.PNG)
![Figure 6.2 Histogram of Toronto Apartment Rooms](./images/bed_bath_den.PNG)

*Figure 6.1 Histogram of Toronto Apartment Proximity to Downtown. Figure 6.2 Histogram of Toronto Apartment Rooms*

We can see from figure 6.1 that the majority of our apartments are in downtown, and are very close to city hall. From Figure 6.2 we see that most of our apartment are single bedroom, single bathroom, and without a den. That makes sense for most downtown apartments. Lets plot the two against each other and see what kind of trend we get.

![Figure 7 Toronto Apartment Price vs. Downtown Proximity](./images/downtown_prox_vs_price.PNG)

*Figure 7 Toronto Apartment Price vs. Downtown Proximity*

Next I am looking at the bedrooms, bathrooms and dens to identify correlations and trends without introducing our additional venue data. Lets use the same plot as above to see if those characteristics can help us identify different price brackets close to downtown. 

![Figure 8. Single Bedroom Apartments: Price vs Proximity to Downtown](./images/single_bedroom_price.PNG)

*Figure 8. Single Bedroom Apartments: Price vs Proximity to Downtown*

![Figure 9. Single Bathroom Apartments: Price vs Proximity to Downtown](./images/single_bathroom_price.PNG)

*Figure 9. Single Bathroom Apartments: Price vs Proximity to Downtown*

![Figure 10. Apartments without a Den: Price vs Proximity to Downtown](./images/no_den_price.PNG)

*Figure 10. Apartments without a Den: Price vs Proximity to Downtown*

We see from these three figures that the prices look somewhat linear right of 0.4 scaled distance units, however everything from 0 - 0.4 units is nowhere near a linear trend. Even combining these attributes, we are going to have a lot of error with a regression model unless we introduce more data. This is where the venue data will help us to get better predictions.

### 3.3 Use of Machine Learning
An array of regression algorithms were used to benchmark the performance on the dataset. I started with simple linear algorithms, and worked my way up to ensemble methods like Random Forest and Gradient Boosted Decision Trees. Each model was evaluated by taking the mean squared error of the price prediction.

![Figure 11. Benchmarking Machine Learning Algorithms](./images/model%20evaluation.PNG)

*Figure 11. Benchmarking Machine Learning Algorithms*

The elastic net performs the best out of the box, so that is the model we will select. It has a root mean square error (RMSE) of $432 and a R-squared correlation of 61% on the test set. 

![Figure 12. Benchmarking Machine Learning Algorithms](./images/model_results.PNG)

*Figure 12. Benchmarking Machine Learning Algorithms*

Next I tuned the model hyperparameters to make the model the best it can be, evaluating its improvement with the test set. I started by doing a random search of the parameters to find a good guess for each, and then used grid search to find the best parameters starting with those guesses. After hyperparameter tuning we get the test set RMSE down to $299, with an accuracy of 86%! Finally we test our tuned model on the validation set and got 74% accuracy with a MSE of $341.

![Figure 13. Benchmarking Machine Learning Algorithms](./images/model_results.PNG)

*Figure 13. Benchmarking Machine Learning Algorithms*


[To Top](#table-of-contents)

## 4 - Results
Results section where you discuss the results.

[To Top](#table-of-contents)

## 5 - Discussion 
Discussion section where you discuss any observations you noted and any recommendations you can make based on the results.

[To Top](#table-of-contents)

## 6 - Conclusion 
Conclusion section where you conclude the report.

[To Top](#table-of-contents)

## 7 - References

1. Toronto apartment prices dataset: *https://www.kaggle.com/rajacsp/toronto-apartment-price*

2. Toronto borough, and neighborhood for each postal code: [Clustering and Segmenting Toronto Neighborhoods.ipynb](./Clustering%20and%20Segmenting%20Toronto%20Neighborhoods.ipynb)

[To Top](#table-of-contents)
