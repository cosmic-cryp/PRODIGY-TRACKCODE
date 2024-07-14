import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data=pd.read_csv(r"C:\Users\Patoju Karthikeya\Downloads\smart_home_device_usage_data.csv")
data.head(10)
UserID	DeviceType	UsageHoursPerDay	EnergyConsumption	UserPreferences	MalfunctionIncidents	DeviceAgeMonths	SmartHomeEfficiency
0	1	Smart Speaker	15.307188	1.961607	1	4	36	1
1	2	Camera	19.973343	8.610689	1	0	29	1
2	3	Security System	18.911535	2.651777	1	0	20	1
3	4	Camera	7.011127	2.341653	0	3	15	0
4	5	Camera	22.610684	4.859069	1	3	36	1
5	6	Thermostat	3.422127	5.038625	1	0	3	1
6	7	Security System	21.065640	2.229344	0	0	56	0
7	8	Security System	23.317096	2.791421	0	0	53	0
8	9	Security System	4.663108	1.780082	1	2	23	1
9	10	Camera	17.468553	7.212756	1	4	58	0
y = data["SmartHomeEfficiency"]
x = data.drop(["SmartHomeEfficiency", "DeviceType"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
LinearRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse*100}")
Mean Squared Error: 11.484650293266444
y_pred
array([ 0.46389663,  0.34615507,  0.18927667, ..., -0.03434992,
        0.07020831,  0.60187366])
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5403 entries, 0 to 5402
Data columns (total 8 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   UserID                5403 non-null   int64  
 1   DeviceType            5403 non-null   object 
 2   UsageHoursPerDay      5403 non-null   float64
 3   EnergyConsumption     5403 non-null   float64
 4   UserPreferences       5403 non-null   int64  
 5   MalfunctionIncidents  5403 non-null   int64  
 6   DeviceAgeMonths       5403 non-null   int64  
 7   SmartHomeEfficiency   5403 non-null   int64  
dtypes: float64(2), int64(5), object(1)
memory usage: 337.8+ KB
new_data = pd.DataFrame({"UserID":[5],"UsageHoursPerDay": [2000], "EnergyConsumption": [3], "UserPreferences": [2],"MalfunctionIncidents":[4],"DeviceAgeMonths":[36]})
predicted_price = model.predict(new_data)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
