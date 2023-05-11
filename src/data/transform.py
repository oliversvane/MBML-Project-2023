import pandas as pd
from meteostat import Point, Daily
from typing import Optional

def prep_data(df:pd.DataFrame,ariport_data_path:str, departure_airport:Optional[list]=None) -> pd.DataFrame:
    """
    Preprocesses the data to a standart timeseries format dataframe. The function uses external weather data to enrich the data.

    Parameters:
    --------------
    df: pd.DataFrame
        Raw dataframe directly from source
    airport_data_path: str
        path to airport data
    departure_airport: list [Optional]
        List of sources to include in the resulting airport


    Returns:
    --------------
    A time series dataframe
    """
    
    col_list =['FlightDate','Origin','DepDelayMinutes'] # Columns to include in putput
    
    if departure_airport != None:
        df = df[df['Origin'].isin(departure_airport)]
    else:
        departure_airport = df['Origin'].unique()

    df = df[col_list]

    df['FlightCount'] = 1
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])
    start_date = df['FlightDate'].min()
    end_date = df['FlightDate'].max()

    #group data
    df = df.groupby(['FlightDate','Origin']).agg({'DepDelayMinutes': 'mean', 'FlightCount': 'sum'})
    #add weather data
    airport_data = pd.read_csv(ariport_data_path)
    weather_data = pd.DataFrame()
    for airport in departure_airport:
        point = airport_data.loc[airport_data['code'] == airport, 'location'].values[0]
        point_list = point.split("(")[-1].split(")")[0].split(" ")
        weather_data_ = Daily(Point(float(point_list[1]), float(point_list[0])), start_date, end_date).fetch()
        weather_data_ = weather_data_.drop(['snow', 'wpgt', 'tsun'], axis=1)
        weather_data_["Origin"] = airport
        weather_data_ = weather_data_.reset_index()
        weather_data_.rename(columns={'time': 'FlightDate'}, inplace=True)
        weather_data = pd.concat([weather_data, weather_data_], axis=0)
      
    df = df.reset_index()
    df = pd.merge(weather_data, df,on=['FlightDate','Origin'], how='left')
    df['DepDelayMinutes'].fillna(0)
    df['FlightCount'].fillna(0)
    

    df['day_of_week'] = [i.weekday() for i in df.FlightDate]
    df['is_weekend'] =  [1 if ((i == 5) | (i == 6)) else 0 for i in df['day_of_week']]
    
    return df