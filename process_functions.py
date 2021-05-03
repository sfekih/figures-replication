
import numpy as np
import pandas as pd

"-------------------------------- FUNCTIONS USED FOR BOTH P2 AND P4 -----------------------------------"

def drop_aberrant_values(df):
    """
    drops aberrant values in the dataframe:
    i) latitude must be greater than 0° and smaller than 90°
    ii) longitude must be greater than -180° and smaller than 180°
    drop rows where these criteria aren't satisfied
    """
    return df[(df['latitude']>0) & (df['latitude']<90) & (df['longitude']>-180) & (df['longitude']<180)]

def drop_inf_na(df):
    """
    drops +inf , -inf , nan values from dataset
    """
    df.replace ([np.inf, -np.inf], np.nan)
    return df.dropna()

def clean_data(df):
    """
    function to clean data:
    i) drop inf,na values
    ii) drop aberrant values in longitude and latitude
    """
    return drop_aberrant_values(drop_inf_na(df))
          

def calculate_distance (lat1,lon1,lat2,lon2):
    """
    R : radius of earth : 6378.137 km
    lat1,lon1 : latitude and longitude of one user
    lat2,lon2 : latitude and longitude of other user
    """
    R = 6378.137
    # convert into radians
    lat1_rad=np.deg2rad(lat1)
    lat2_rad=np.deg2rad(lat2)
    lon1_rad=np.deg2rad(lon1)
    lon2_rad=np.deg2rad(lon2)
    
    #get difference of lattitude and difference of longitude
    delta_lat=lat2_rad-lat1_rad
    delta_lon=lon2_rad-lon1_rad
    
    #return formula Haversine formula
    a=((np.sin(0.5*delta_lat))**2)+np.cos(lat1_rad)*np.cos(lat2_rad)*((np.sin(0.5*delta_lon))**2)
    return 2*R*np.arcsin(np.sqrt(a))

def find_homes (df):
    """
    function created to find the home of each user : 
    i) divides the world into 25x25km cells
    ii) finds square where most checkins
    iii) home adress is the mean of the locations of square with most visits
    """
    #set origin point ( can be anything , doesn't matter)
    lat_min=0
    long_min=0
    
    df1=df.copy()
    
    #for each user : report the square in which each checkin - was
    df1['lat']=((calculate_distance(df['latitude'],0,lat_min,0)/25)).astype(int) 
    df1['lon']=((calculate_distance(0,df['longitude'],0,long_min)/25)).astype(int) 
    
    #For each user : 
    # i) calculate mean latitude and longitude in each square
    # ii) get number of checkins in each square
    df_grouped=df1.groupby(['User','lon','lat'],as_index=False).agg({
        'longitude' : 'mean',
        'latitude' : 'mean',
        'location id' : 'size'
    })
    
    #get indices of square where most indices
    #variable max_indexes created just for clarity reasons
    max_indices=df_grouped.groupby('User')['location id'].idxmax()
    return df_grouped.loc[max_indices]

"---------------------------- FUNCTIONS USED JUST FOR P4 ----------------------------------------"
################# FUNCTIONS FOR PROCESSING DATA ########################
def Bool_visited_friend(x):
    """
    Function that returns a boolean : 
    OUTPUT:
    i) True if user checked in at least one friend's house for a given distance from home
    ii) False if user didn t check in any friend's house
    """
    sum_=sum(x)
    if sum_==0 : return 0
    else : return 1


def get_dist_from_home (CHECKIN_PATH,names_checkin):
    """
    function to find the distance each user went from his home
    To do that, we find the home location of the user then compute the distance between the checkin location and the home
    Working with radius of 25km, we will approximate each value that is inferior to 25km to 25km

    We import here checkin dataset because we won't need it after that so we don't need to keep it in memory
    """
    #import checkin dataset
    checkin_df=pd.read_csv(CHECKIN_PATH,compression='gzip',delimiter='\t',names=names_checkin,index_col=None)
    checkin_df=clean_data(checkin_df).sample(frac=0.4)

    #find the home of each user
    homes_df=find_homes(checkin_df)[['User','longitude','latitude']].\
            rename(columns={'longitude': 'longitude home x',\
                            'latitude': 'latitude home x',\
                           'User':'User x'})
    
    #merge the dataset that only contains the home of each user with the original dataset
    homes_checkins_df=checkin_df[['User','longitude','latitude']].\
            rename(columns={'longitude': 'checkin longitude x',\
                            'latitude': 'checkin latitude x',\
                            'User':'User x'}).\
            merge(homes_df)
    
    #Now we calculate the distance between the checkin position and the home
    homes_checkins_df['distance from home x(km)']=calculate_distance(homes_checkins_df['checkin latitude x'],\
            homes_checkins_df['checkin longitude x'], homes_checkins_df['latitude home x'],\
            homes_checkins_df['longitude home x']).dropna().round()
    
    #The paper only keeps distances that are >25km, anything less than 25km is considered to be 25km
    homes_checkins_df.loc[homes_checkins_df['distance from home x(km)']<25,'distance from home x(km)']=25

    return homes_df,homes_checkins_df[['User x','checkin longitude x','checkin latitude x','distance from home x(km)']]

def apply_median(y,N=6):
    """
    smoth curve using median
    y : to be plotted
    N : Number of items to use to smooth curve
    We choose not to do this process for small values because the curve is already smooth for small distances
    """
    y1=np.copy(y)
    
    for i in range (N,len(y)):
        if i>N:y1[i]=np.median(y[0:i+N])
        elif i<len(y)-N : y1[i]=np.median(y[i-N:i+N])
        else : y1[i]=np.median(y[i-N:len(y)])
    
    return y1

def get_approximation (x_original,y_original,x_log,kind='slinear',N=5):
    """
    INPUTS:
    x_original : original x axis (the one that contains all the distances)
    y_original : original y axis (the one that contains all the output)
    x_log : x with a logscale
    kind : kind of interpolation we want to do
    N : number of points used for moving median
    OUTPUTS:
    clean vector for plot
    """
    f=interp1d(x_original,y_original,kind='slinear')
    return apply_median(f(x_log),N)

"---------------------------- FUNCTIONS BELOW USED JUST FIGURE 2A ----------------------------------------"


def nb_unidimentional_arrays(df):
    """
    counts numbers of unidimentional arrays in edges dataset
    """
    df1=df.copy()
    df1['sorted']=df1.min(axis=1).astype(str)+','+df1.max(axis=1).astype(str)

    #group by sorted and count number of rows for each different output in sorted
    grouped = df1.groupby('sorted')['followed'].count()

    #get indices where it's unidimentional
    uni_indices=[i for i in range (np.shape(grouped)[0]) if grouped[i]!=2]

    return len(uni_indices)


def plotting_df(df):
    '''
    function to create new dataset having only two columns : distance and its probability
    This dataset returned will be used to plot
    '''
    df1=df.copy()
    
    #let only distances > 0 because we think distance = 0 means it s an error in the data
    df1=df1[df1['distance']>0]
    
    #each distance was calculated twice due to the fact that edges are bidimentional so we drop duplicates
    df1=df1.drop_duplicates('distance')
    
    #sort distances ascendingly 
    df1=df1.sort_values(by=['distance'])
    
    #drop NA values if there are any and round values of distances
    df1=df1.dropna().round()
    
    #rename : nothing important, just for clarity
    df1=df1.rename(columns={'latitude_followed': 'probability'})
    
    ###get probabilities :
    # i ) count number of times each distances  was repeated
    df1=df1.groupby('distance',as_index=False)['probability'].count()
    # ii) divide by total number 
    df1['probability']=df1['probability']/df1['probability'].sum()
    return df1

def get_distances(edges_df,checkin_df):
    """
    this function :
    
    i) gets the adress of each user (calls the function find_homes)
    
    ii) returns one dataset that returns for each line : user following and his adress,
    user followed and his adress, the distance between the two users 
    (its only use is to check the sanity of results)
    Note that user following and user followed can be inverted since the edges are bidirectioned.
    
    iii) returns another dataset, containing the distances and the probability of each one 
    (used for the plot later)
    """
    #get adress of each user
    proceeded_df=find_homes(checkin_df)
    
    #merge checkin dataset with column 'following'of edges dataset
    df=pd.merge(edges_df,proceeded_df,left_on='following', right_on='User')
    
    #rename to have meaningful output
    df=df.rename(columns={'latitude': 'latitude_following','longitude': 'longitude_following'})
    
    #merge dataset with column 'followed' of edges dataset
    df=pd.merge(df,proceeded_df,left_on='followed', right_on='User')
    
    #drop meaningless columns
    df=df.drop(['lon_x', 'lat_x','lon_y','lat_y','location id_x',                'location id_y','User_x','User_y'], axis=1)
    #rename to have meaningful output
    df=df.rename(columns={'latitude': 'latitude_followed','longitude': 'longitude_followed'})
    
    #calculate distance between homes
    df['distance']=calculate_distance(df['latitude_following'],df['longitude_following'],                        df['latitude_followed'],df['longitude_followed'])
    
    return df,plotting_df(df)

