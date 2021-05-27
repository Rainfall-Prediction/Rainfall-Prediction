# Rainfall Prediction

## Team Members:
1) Ashwin Biju Alikkal
2) Anshika Maharana
3) Ashutosh
4) Simran Kaur

## Domain: Environment

## Aim:
To apply various Machine Learning models on the rainfall dataset to predict whether it is going to rain or not the following day.

## Dataset Description:
This dataset contains about 11 years (2007-2017) of daily weather observations from many locations across Australia. The Data has been processed to provide a target variable "RainTomorrow" (whether it rains on the following day - Yes/No).

* Data source: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
* An example of latest weather observations in Canberra: http://www.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml
* Features description at https://www.rdocumentation.org/packages/rattle/versions/5.4.0/topics/weather
* Definitions adapted from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml

## 1. Data Cleaning:
### 1.1 Handling Missing Data:
For handling the missing values (NaN) in our data, we adopted two approaches.

**1.1(1) Approach 1st : Median/Mode Imputation**

In this approach, we imputed all the NaN values in the _continous features_ using the median with respect to a particular month. For example, for a NaN value present in the feature _"MinTemp"_, corresponding to "Month 5", median of the values of _"MinTemp"_ for the "Month 5" was used to replace it. Similarly, for all the NaN values present in the _categorical features_, we used mode for the imputation with respect to different locations. 

**1.1(2) Approach 2nd : MICE Imputation**

_Multiple Imputation by Chained Equations (MICE)_ is a robust, informative method of dealing with missing data in datasets. The procedure ‘fills in’ (imputes) missing data in a dataset through an iterative series of predictive models. In each iteration, each specified variable in the dataset is imputed using the other variables in the dataset. 



