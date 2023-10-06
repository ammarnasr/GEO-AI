import os
from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    MimeType,
    DataCollection,
    bbox_to_dimensions,
    SentinelHubRequest,
)
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import requests
import jsonlines as jsonl



def get_bounds_of_polygon(polygon):
    bounds = polygon.bounds
    bounds = [bounds[0], bounds[1], bounds[2], bounds[3]]
    return bounds


def get_final_dir(location_name, date, evalscript):
    data_dir = './data/satellite_images'
    os.makedirs(data_dir, exist_ok=True)
    target_date_dir = os.path.join(data_dir, date)
    os.makedirs(target_date_dir, exist_ok=True)
    location_dir = os.path.join(target_date_dir, location_name)
    os.makedirs(location_dir, exist_ok=True)
    evalscript_dir = os.path.join(location_dir, evalscript)
    os.makedirs(evalscript_dir, exist_ok=True)
    final_dir = evalscript_dir
    return final_dir



def get_any_image_from_sentinelhub(polygon, date, evalscript, location):
    final_dir = get_final_dir(location, date, evalscript)
    bbox = get_bounds_of_polygon(polygon)
    evalscript_api = get_sentinelhub_api_evalscript(evalscript)
    config = get_sentinelhub_api_config()
    sen_obj = SenHub(config)
    sen_obj.set_dir(final_dir)
    sen_obj.make_bbox(bbox)
    sen_obj.make_request(evalscript_api, date)
    imgs = sen_obj.download_data()
    return imgs[0], final_dir


def get_sentinelhub_api_config():
    config = SHConfig()
    config.instance_id       = '44c8ea2e-d9ca-4fc1-9326-c9a95d98225e'   
    config.sh_client_id      = '8955b169-0ae5-4775-a68d-c1bcf0f95310' 
    config.sh_client_secret  = '@12~GlPfQhTxT^<?V8kw1HXo52IfcS_C*_2:5t(I'
    return config


def get_sentinelhub_api_token():
    config = get_sentinelhub_api_config()
    token = SenHub(config).token
    return token




def get_available_dates_from_sentinelhub(bbox, token, start_date, end_date):
    '''
    Get a list of dates that have available images for a specific bounding box and time period
    from the SentinelHub API
    args:
        bbox: the bounding box of the area of interest
        token: the SentinelHub API token
        start_date: the start date of the time period
        end_date: the end date of the time period
    return:
        dates: a list of dates that have available images
    '''
    dates = get_cached_available_dates_from_sentinelhub(bbox, start_date, end_date)
    if dates is not None:
        print('dates fetched from cache')
        return dates
    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer '+ token,
    }
    data = f'{{ "collections": [ "sentinel-2-l2a" ], "datetime": "{start_date}T00:00:00Z/{end_date}T23:59:59Z", "bbox": {bbox}, "limit": 100, "distinct": "date" }}'
    response = requests.post('https://services.sentinel-hub.com/api/v1/catalog/search', headers=headers, data=data)
    dates = response.json()['features']
    cache_available_dates_from_sentinelhub(bbox, start_date, end_date, dates)
    print('dates fetched from api')
    return dates



def get_sentinelhub_api_evalscript(script_name='TRUECOLOR', info=False):
    '''
    Get a SentinelHub API evalscript based on the name of the script
    args:
        script_name: the name of the script
    return:
        evalscript: a SentinelHub API evalscript
    '''
    if info:
        print(f'''
              The following scripts are available:
                    CAB: Chlorophyll Absorption Band Index
                    FCOVER: Fractional Cover
                    LAI: Leaf Area Index
                    TRUECOLOR: True Color
                    CLP: Cloud Probability
                    ALL: All Indices
                    NDVI: Normalized Difference Vegetation Index
                ''')  
    
    # Open and Read The Javascripts that will be passed to the SentinelHub API
    if os.path.exists('./scripts/cab.js'):
        with open('./scripts/cab.js') as f:
            evalscript_cab = f.read()
    else:
        evalscript_cab = None

    if os.path.exists('./scripts/fcover.js'):
        with open('./scripts/fcover.js') as f:
            evalscript_fcover = f.read()
    else:
        evalscript_fcover = None

    if os.path.exists('./scripts/lai.js'):
        with open('./scripts/lai.js') as f:
            evalscript_lai = f.read()
    else:
        evalscript_lai = None

    if os.path.exists('./scripts/truecolor.js'):
        with open('./scripts/truecolor.js') as f:
            evalscript_truecolor = f.read()
    else:
        evalscript_truecolor = None

    if os.path.exists('./scripts/clp.js'):
        with open('./scripts/clp.js') as f:
            evalscript_clp = f.read()
    else:
        evalscript_clp = None

    if os.path.exists('./scripts/all.js'):
        with open('./scripts/all.js') as f:
            evalscript_all = f.read()
    else:
        evalscript_all = None

    if os.path.exists('./scripts/ndvi.js'):
        with open('./scripts/ndvi.js') as f:
            evalscript_ndvi = f.read()
    else:
        evalscript_ndvi = None

    # Dictionry of JavaScript files
    Scripts = {
        'CAB': evalscript_cab,
        'FCOVER': evalscript_fcover,
        'LAI': evalscript_lai,
        'TRUECOLOR': evalscript_truecolor,
        'CLP': evalscript_clp,
        'ALL': evalscript_all,
        'NDVI': evalscript_ndvi
    }
    
    if script_name in Scripts.keys():
        return Scripts[script_name]
    else:
        keys = Scripts.keys()
        keys = list(keys)
        print(f'Script name must be one of the following: {keys}')
        return None


class SenHub:
    ''' 
    Class For handling requests to Senhub API.
    '''
    def __init__(self,config,  resolution = 10,
                data_source = DataCollection.SENTINEL2_L1C,
                identifier ='default', mime_type = MimeType.TIFF):
        self.resolution = resolution
        self.config = config
        self.setInputParameters(data_source)
        self.setOutputParameters(identifier, mime_type)
        self.set_token()

    def setInputParameters(self, data_source):
        '''
        Select Source Satellite 
        '''
        self.data_source = data_source
    
    def setOutputParameters(self,identifier, mime_type):
        '''
        Select The return Type of request format and identifier
        '''
        self.identifier = identifier
        self.mime_type = mime_type

    def set_token(self):
        '''
        Fetch Tooken from sentinelhub api to be used for available dates 
        '''
        client_id = self.config.sh_client_id
        client_secret = self.config.sh_client_secret
        client = BackendApplicationClient(client_id=client_id)
        oauth = OAuth2Session(client=client)
        token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/oauth/token',client_secret=client_secret)
        self.token =  token['access_token']

    def get_input_data(self, date):
        '''
        Wrap input_data to provide to the sentinelhub API
        '''
        return SentinelHubRequest.input_data(data_collection=self.data_source, time_interval=(date, date))

    def get_output_data(self):
        '''
        Wrap output_data to provide to the sentinelhub API
        '''
        return SentinelHubRequest.output_response(self.identifier, self.mime_type)
        
    def set_dir(self, dir_path):
        '''
        Set The Tragt Download Directory Path
        '''
        self.dir_path = dir_path

    def make_bbox(self, bbox):
        '''
        Wrap bbox to provide to the sentinelhub API.
        '''
        self.bbox = BBox(bbox=bbox, crs=CRS.WGS84)
        self.bbox_size = bbox_to_dimensions(self.bbox, resolution=self.resolution)
                
    def make_request(self, metric, date):
        '''
        Setup the Sentinal Hub Request
        '''
        input_data = self.get_input_data(date)
        output_data = self.get_output_data()
        self.request = SentinelHubRequest(
            data_folder=self.dir_path,
            evalscript=metric,
            input_data=[input_data],
            responses=[output_data],
            bbox=self.bbox,
            size=self.bbox_size,
            config=self.config,
            )

    def download_data(self, save=True , redownload=False):
        '''
        Make The Request and download the data
        '''
        return self.request.get_data(save_data=save, redownload=redownload)





def cache_available_dates_from_sentinelhub(bbox, start_date, end_date, dates):
    '''
    Cache a list of dates that have already been fetched from the SentinelHub API.
    This is to avoid making repeated requests to the API. The cached dates are stored
    in a jsonl file called cached_dates.jsonl in a folder called cache/cached_dates.
    args:
        bbox: the bounding box of the area of interest
        start_date: the start date of the time period
        end_date: the end date of the time period
        dates: a list of dates that have available images
    return:
        None
    '''
    cache_folder = './cache'
    os.makedirs(cache_folder, exist_ok=True)
    cache_dates_folder = os.path.join(cache_folder, 'cached_dates')
    os.makedirs(cache_dates_folder, exist_ok=True)
    cache_dates_file = os.path.join(cache_dates_folder, 'cached_dates.jsonl')
    if not os.path.exists(cache_dates_file):
        with open(cache_dates_file, 'w') as f:
            f.write('')
    current_entry = {
        'bbox': bbox,
        'start_date': start_date,
        'end_date': end_date,
        'dates': dates
    }
    with jsonl.open(cache_dates_file, mode='a') as writer:
        writer.write(current_entry)


def get_cached_available_dates_from_sentinelhub(bbox, start_date, end_date):
    '''
    Get a list of dates that have available images for a specific bounding box and time period
    that have already been fetched from the SentinelHub API. if the dates have not been fetched
    before, return None.
    args:
        bbox: the bounding box of the area of interest
        start_date: the start date of the time period
        end_date: the end date of the time period
    return:
        dates: a list of dates that have available images
    '''
    cache_dates_file = './cache/cached_dates/cached_dates.jsonl'
    if not os.path.exists(cache_dates_file):
        return None
    current_entry = {
        'bbox': bbox,
        'start_date': start_date,
        'end_date': end_date,
        'dates': []
    }
    with jsonl.open(cache_dates_file, mode='r') as reader:
        for entry in reader:
            if entry['bbox'] == bbox and entry['start_date'] == start_date and entry['end_date'] == end_date:
                return entry['dates']
    return None

