# Rainfall Prediction

### Team Members:
1) Ashwin Biju Alikkal
2) Anshika Maharana
3) Ashutosh
4) Simran Kaur

### Domain: Environment

### Objective:
To apply various Classification and Regression machine learning models on the rainfall dataset to predict whether it is going to rain or not the following day. And if yes, then how much mm ?

### Dataset Description:
This dataset contains about 11 years (2007-2017) of daily weather observations from many locations across Australia. The Data has been processed to provide target variables "RainTomorrow" (whether it rains on the following day - Yes/No) and “Rainfall”.

* Data source: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
* An example of latest weather observations in Canberra: http://www.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml
* Features description at https://www.rdocumentation.org/packages/rattle/versions/5.4.0/topics/weather
* Definitions adapted from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml

### A basic outline
This section contains a basic layout of what do we plan to do further in the project and how.

##### 1. Data Cleaning and Pre-processing:
In this module, we plan to work on understanding the data better. For this we research about the features given to us and try to observe the trends in our data to the actual weather conditions in the given locations of Australia during different seasons.
Therefore, we try to find a way to impute missing values in the data based on our observations about the data and handle the impact of outliers.
If everything goes smoothly, we plan to cover this within this fortnight.

##### 2. Exploratory Data Analysis (EDA):
In this module, we try to convey our understanding and observations on the data via graphs and plots. We also try to understand what kind of a model the data follows.
If everything goes smoothly, we plan to cover this within this fortnight.

##### 3. Feature Engineering:
In this module, we try to prepare a proper input dataset, compatible with the ML algorithms to improve the performance of the ML models. This includes Imputation, handling outliers, binning, transforming features, encoding, scaling and normalizing and balancing the target variable.

##### 4. Model Building & Evaluation:
Here, we try to find answers to two main questions…”Will it Rain Tomorrow? If yes then how much?”. We fit different ML models to our dataset to find the answers to these questions as close as possible. Clearly, we’ll be applying both Classification as well as Regression algorithms to know the answers.

### Understanding the data and its features
##### 1. "Date"
•	We have the data for weather starting from November 2007 to June 2017.

##### 2. "Location"
•	We have data on total of 49 different locations throughout Australia.
•	For each month in each year, we have the data on different no. of locations.
•	In 2007, we have the data on only one location and only on few location in the year 2008.

##### 3. "MinTemp" and "MaxTemp"
•	The minimum and maximum temperature in Degrees Celsius recorded on a given day respectively.
•	On plotting the histogram, we can see an almost normal curve.
•	The data follows an almost similar pattern in each month but does not follow a similar pattern if plotted for each location.
•	On plotting “MinTemp” and “MaxTemp” with respect to year for each location, we noticed that in most of the cases both the temperature starts decreasing, reaches its lowest value at 2012 and then increases.
•	Strong positive correlation (0.74) between these two features.

##### 4."Rainfall"
•	Precipitation in the 24 hours to 9am in mm.
•	Highly Positively Skewed. (9.88).

##### 5. "Evaporation"
•	"Class A" pan evaporation in the 24 hours to 9am in mm.
•	Has around 42% of the data missing.
•	We don’t have the data on Evaporation for 16 locations.
•	Highly Positively Skewed. (3.74)

##### 6. "Sunshine"
•	Bright sunshine in the 24 hours to midnight.
•	Has around 47% of the data missing.
•	We don’t have the data on Sunshine for 19 locations.
•	Negatively Skewed.

##### 7. "WindGustDir"
•	Direction of strongest gust in the 24 hours to midnight.
•	No data for ‘Albany’ and ‘Newcastle’.

##### 8. "WindGustSpeed"
•	Speed of strongest wind gust in the 24 hours to midnight. (Km/hr)
•	No data for ‘Albany’ and ‘Newcastle’.
•	Positively Skewed.

##### 9. "WindDir9am" and "WindDir3pm"
•	Wind direction averaged over 10 minutes prior to 9am and 3pm respectively.
•	For ‘Watsonia’ we have two modes in WindDir3pm.

##### 10. "WindSpeed9am" and "WindSpeed3pm"
•	Wind speed (km/hr) averaged over 10 minutes prior to 9am and 3pm respectively.
•	Both features are positively skewed.
•	WindSpeed3pm is positively correlated with WindGustSpeed.(0.69)

##### 11. "Humidity9am" and "Humidity3pm"
•	This feature tells us about the relative humidity at 9am and 3pm respectively.
•	Relative humidity tells us how much water vapour is in the air, compared to how much it could hold at that temperature. It is shown as a percent. For example, a relative humidity of 50 percent means the air is holding one half of the water vapour it can hold.
•	“Humidity9am” is negatively skewed.
•	“Humidity3pm” has an almost normal curve.
•	Humidity for all the locations in every month behaves similarly for both the timings.

##### 12. "Pressure9am" and "Pressure3pm"
•	These variables measure the atmospheric pressure at 9am and 3pm respectively observed at the mean sea level. The values are given in hectopascals (or millibar). Hectopascals is the SI unit for measuring atmospheric pressure.
•	Data for same 4 location is missing in both the features.
•	On plotting the data for these features overall, month wise and location wise, a normal curve was observed.
•	Very strong positive correlation (0.96) between these two features.

##### 13. "Cloud9am" and "Cloud3pm"
•	Has around 37% and 40% of the data missing respectively.
•	Fraction of sky obscured by clouds at 9 am and 3pm respectively. This is measured in 'oktas' which are a unit of eights. It records how many eights of the sky are obscured by cloud. A 0 measure indicates completely clear sky whilst an 8 indicates that it is completely overcast.
•	Data for same 12 locations is missing in both the features.
•	Both the features are negatively correlated with “Sunshine”. (-0.68 and -0.7 respectively)

##### 14. "Temp9am" and "Temp3pm"
•	Temperature recorded in Degree Celsius at 9am and 3pm respectively.
•	These two variables behaved very similar to the variables “MinTemp” and “MaxTemp”.
•	On plotting these variables for each month we obained an almost normal curve for each month but different distribution, when plotted against locations.
•	Average value of Temp3am always came out to be greater than Temp9am.
•	Very strong positive correaltion (0.86) between these two features.
•	MinTemp is strongly correlated with Temp9am (0.9) and Temp3pm (0.71).
•	MaxTemp is strongly correlated with Temp9am (0.89) and Temp3pm (0.98).

##### 15. "Risk_MM"
•	The amount of next day rain in mm.
•	The “Rainfall” recorded for a day is taken as a risk for the following day.
•	The NaN values in the “Rainfall” feature are recorded as 0 in “RiskMM”.

##### 16. "RainToday" and "RainTomorrow"
•	Yes or No.

##### NOTES :
1.	All the temperature features are strongly correlated with each other.
2.	The features “Sunshine”,”Evaporation”,”Cloud9am” and “Cloud3pm” has the most no. of missing values.
3.	Data for 10 common locations is missing in the features “Sunshine”,”Evaporation”,”Cloud9am” and “Cloud3pm”.
4.	For “Rainfall”>1mm, ‘Yes’ in “RainToday”.
5.	The proportion of ‘No’ is far greater than ‘Yes’ in our target variable “RainTomorrow”.









