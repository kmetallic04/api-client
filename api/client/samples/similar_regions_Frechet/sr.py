import os
import tempfile
import pickle
from datetime import date

import numpy as np
from scipy.linalg import sqrtm
import pandas as pd
from tqdm import tqdm

from sklearn.neighbors import BallTree, DistanceMetric
from groclient.client import GroClient, BatchError
from groclient.lib import get_default_logger
#from api.client.samples.similar_regions.similar_region_state import SimilarRegionState
from sklearn.metrics.pairwise import euclidean_distances

""" API Config """
API_HOST = 'api.gro-intelligence.com'
ACCESS_TOKEN = os.environ['GROAPI_TOKEN']

REGION_INFO_CACHE = "region_info.pickle"
PICKLE_PROTOCOL = 4
REGIONS_PER_QUERY = 100
MAX_RETRY_FAILED_REGIONS = 3
OK_TO_PROCEED_REGION_FRACTION = 0.1 # we want at least 10% of all desired region present in the final data matrix

##############################
# These were constructor parameters but changes to global constants to reduce number of options
#
# Into how many intervals to split calendar year when generating distributions from raw data.
# Default is 52 (roughly weekly). High values (such as 365 - daily) lead to greater storage requirements and not recommended
T_INT_PER_YEAR = 52
# If 'any_missing', drop all regions which do nor have full set of valid datapoints
# (i.e. t_int_per_year data points for each property). If 'fully_missing', drop region only if there is less than 2 valid datapoints
DROP_MODE = 'any_missing'
# If True, reload region info from Gro, otherwise try to use cached (still reload if region not found).
REGION_INFO_RELOAD = False

class SimilarRegion(object):

    def __init__(self, metric_properties,
                 data_dir="similar_regions_cache",
                 metric_weights=None):
        """Initialize the object. No data is downloaded at this stage, only
        when build is called separately.

        :param metric_properties: A dict containing properties which
        can be used in the region similarity metric.  This is by
        default defined in metric.py, but can be adjusted.

        :param data_dir: directory to use as a data cache. If not
        specified, a temp dir is created.

        :param metric_weights: A {property: weight} dictionary stating
        specific properties and their weights to use for this run. The
        properties must be a subset of those in metric_properties.  If
        not provided, all properties from metric_properties are used
        with equal weights.

        """
        self.t_int_per_year = T_INT_PER_YEAR
        self.t_int_days = 365 / self.t_int_per_year # number of days in single t interval (can be float)

        self._logger = get_default_logger()

        # extract needed parameters from inputs
        self.metric_properties = metric_properties
        if metric_weights is not None:
            unresolved_items = [item for item in metric_weights if item not in self.metric_properties]
            assert not unresolved_items, "Found items {} in metric instance not present in metric description".format(unresolved_items)
            self.metric_weights = metric_weights
        else:
            self.metric_weights = dict(zip(self.metric_properties.keys(),[1.0]*len(self.metric_properties)))
        self.needed_properties = sorted(self.metric_weights.keys())

        if data_dir:
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
            self.data_dir = data_dir
        else:
            self.data_dir = tempfile.gettempdir()

        self._logger.debug('Using {} as data_dir'.format(self.data_dir))

        self.client = GroClient(API_HOST, ACCESS_TOKEN)
        self.search_region = None

    def build(self, search_region=0):
        self.data = None
        self.to_retry = []
        self.available_regions = []

        self._logger.info("(Re)building similar region object with region {} data_dir {} t_int_per_year {}".format(
            search_region, self.data_dir, self.t_int_per_year))

        self._initialize_regions(search_region)

        self._data_download()

        self._drop_incomplete()

        self._normalize()

        self._reformat()

        ##################### Region similarity metric ##############
        # DistanceMetric allows only specific function signature (copy-pasted from docs):
        # Takes two one-dimensional numpy arrays, and returns a distance.
        # Note that in order to be used within the BallTree, the distance must be a true metric
        # i.e. it must satisfy the following properties
        # 1. Non-negativity: d(x, y) >= 0
        # 2. Identity: d(x, y) = 0 if and only if x == y
        # 3. Symmetry: d(x, y) = d(y, x)
        # 4. Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)
        #
        # So define distance function here to have access to object members
        def dist_function(x,y):
            # means
            x_m = x[:self.n_pit+self.n_ts]
            y_m = y[:self.n_pit+self.n_ts]

            # Frechet (AKA 2-Wasserstein) distance between two distributions
            # first part is simply L2 between means, second is metric on the space of covar matrices
            # Cov matrices formally have zero col/rows corresponding to pit variables
            # these do not contribute to trace, but sqrtm might complain about singular matrices
            # so take only ts part
            # Note: this is inner loop, so maybe explore np.einsum()
            # but for now keep trace/dot/sum for better readability
            m_dist = np.sum((x_m-y_m)**2)
            s_dist = 0
            if self.n_ts > 0:
                # covar matrices (ts only, square n_ts x n_ts)
                x_cov = x[-self.n_ts*self.n_ts:].reshape(self.n_ts,-1)
                y_cov = y[-self.n_ts*self.n_ts:].reshape(self.n_ts,-1)
                # can get negative due to numerical errors when x=y, so clamp to 0
                s_dist = max(0, np.trace( x_cov + y_cov - 2*sqrtm(x_cov.dot(y_cov)) ))
            return np.sqrt(m_dist + s_dist)

        # Finally construct ball tree with this distance
        self._logger.info("Constructing BallTrees...")
        self.metric_object = DistanceMetric.get_metric('pyfunc', **{'func':dist_function})
        self.ball = BallTree(self.data, metric=self.metric_object, leaf_size=2)
        self.balls_on_level = {}
        self.regions_on_level = {}
        # separate ball trees to process requests on givel level
        # shouldn't generally make separate data copies since region_id's are organized with countries < provinces < districts
        # so data for each level should be continuous arrays
        for level in [3,4,5]:
            self.regions_on_level[level] = [r for r in self.available_regions if self.region_info[r]['level']==level]
            level_data = self.data[[self.region_index[r] for r in self.regions_on_level[level]],:]
            self._logger.info(" level {}, {} regions".format(level, len(self.regions_on_level[level])))
            if level_data.size > 0:
                self.balls_on_level[level] = BallTree(level_data, metric=self.metric_object, leaf_size=2)
        self._logger.info("DONE")
        self.search_region = search_region
        return

    def _initialize_regions(self, search_region):
        ################ Dealing with region information (load from cache or API) ################
        self.region_info = {}
        info_path = os.path.join(self.data_dir, REGION_INFO_CACHE)
        if os.path.isfile(info_path):
            self._logger.info("Reading region info from cache {}".format(info_path))
            with open(info_path, 'rb') as f:
                self.region_info = pickle.load(f)
        if REGION_INFO_RELOAD or search_region not in self.region_info:
            ri = [self.client.lookup('regions', search_region)] # include top-level region itself
            for l in range(5,ri[0]['level'],-1):
                self._logger.info("Loading region info for geo level {}".format(l))
                ri += self.client.get_descendant_regions(search_region,descendant_level=l,
                                                              include_historical=False,
                                                              include_details=True)
            #reformat as dict with region_id as key
            self.region_info = dict(zip([item['id'] for item in ri], ri))

            # immediately save region_info
            # might overwrite a large cache if it was storing "parallel" region (cache was Asia but asking for US)
            with open(info_path, 'wb') as f:
                self._logger.info("Saving region info file {}".format(info_path))
                pickle.dump(self.region_info,f)
        else:
            # We might have too many regions in the info cache - this happens if cache was filled for a parent
            # of region currently requested (filled for entire world but we want only US, for example)
            top_info = self.region_info[search_region]
            self._logger.info("Search region is present in cache of size {}".format(len(self.region_info)))
            level = top_info['level'] # can only have 2/3/4/5
            to_include = [search_region]
            ri = {search_region:top_info}
            while level<5:
                to_include = [r for r in [j for r in to_include for j in self.region_info[r]['contains']]
                              if self.region_info.get(r,{'level':-1})['level']==level+1]
                ri.update({r:self.region_info[r] for r in to_include})
                level += 1
            self.region_info = ri
            self._logger.info("Trimmed region info to {} items".format(len(self.region_info)))
        self.needed_regions = sorted(self.region_info.keys())
        return

    def _data_download(self):
        ##################### Data download ################################################
        self.data = np.zeros((len(self.needed_regions)*self.t_int_per_year, len(self.metric_weights)))
        self.data[:] = np.nan
        for (prop_i, prop_name) in enumerate(self.needed_properties):
            self.prop_data = self.data[:,prop_i] # alias to specific column of the data matrix
            prop_path = os.path.join(self.data_dir, prop_name+'_'+str(self.t_int_per_year)+'.npz')
            prop_data_cached = np.array(())
            prop_regions = []
            if os.path.isfile(prop_path):
                self._logger.info("Reading property {} from cache {} ...".format(prop_name, prop_path))
                with np.load(prop_path,allow_pickle=True) as prop_file:
                    prop_data_cached = prop_file['data']
                    prop_regions = list(prop_file['regions'])
                self._logger.info("Merging in cached info ...")
                self._merge_into_prop_data(prop_data_cached, prop_regions)
                self._logger.info("Done")
            missing_regions = sorted(set(self.needed_regions) - set(prop_regions))
            # actual download
            if missing_regions:
                valid_data, valid_regions = self._load_property_for_regions(prop_name, missing_regions)
                if valid_regions:
                    np.savez_compressed(prop_path,
                                        data=np.concatenate([prop_data_cached,valid_data], axis=0),
                                        regions=prop_regions+valid_regions)
                    self._merge_into_prop_data(valid_data, valid_regions)
        return

    def _drop_incomplete(self):
        ################### Drop incomplete regions #################################
        self.data = pd.DataFrame(self.data,
                            index=pd.MultiIndex.from_product([self.needed_regions,range(1,self.t_int_per_year+1)],
                                                             names=['region','year_period']),
                            columns=self.needed_properties)
        # Subjective decision - what to do with missing data?
        # option #1 - drop any region with ANY missing data in any of variables
        # option #2 - drop any region with FULLY missing data in at lest one variable
        # option #3 - fill in missing data with averages
        # #3 is problematic since particular region can be VERY different from global average
        # and doing this more locally is rather complicated (have to use geographic proximity info)
        # Let's just drop all regions which have ANY missing values (i.e. missing soil moisture in Dec
        # is sufficient to be dropped). This is by far the simplest but possibly a bit extreme.
        # as it puts burden on the user to select well-populated variables for his metrics.
        # Might want to revisit later
        self._logger.info("Dropping regions with insufficient data ...")
        if DROP_MODE == 'any_missing':
            # take only those with full count for all properties
            tmp = (self.data.groupby(level='region').count() == self.t_int_per_year).all(axis=1)
        elif DROP_MODE == 'fully_missing':
            # take those where we have at least some valid data, i.e. at least two points with all properties filled
            # (one such point is insufficient for std calculation)
            # will have nan's is the data, so will do dropna before covariance computation
            # some regions will be of lower quality, so use this mode with care
            tmp=(~self.data.isna()).all(axis=1).groupby(level='region').sum() >= 2
            #tmp = (self.data.groupby(level='region').count() > 0).all(axis=1)
        else:
            assert False, "Unknown DROP_MODE"
        self.data = self.data.loc[self.data.index.isin(tmp[tmp].index,level='region')].sort_index()
        self.available_regions = sorted(list(self.data.index.get_level_values('region').unique()))
        self.num_regions = len(self.available_regions)

        self.region_index = dict(zip(self.available_regions,range(self.num_regions)))
        self.prop_index = dict(zip(self.needed_properties,range(len(self.needed_properties))))

        self._logger.info("{} regions remains".format(self.num_regions))
        assert self.num_regions > OK_TO_PROCEED_REGION_FRACTION*len(self.needed_regions), "Less than {}% of desired regions has full data. Bailing out.".format(OK_TO_PROCEED_REGION_FRACTION*100)
        return

    def _normalize(self):
        ################### Normalization/weighting ################
        # If provided with user weight/norm const for given property,
        # use these, otherwise take 1/data std respectively
        # This brings all data to the space over which we will compute distance

        # Note that us taking sqrt of user-provided weight here assumes user wants something like
        # dist^2 = \sum w_user_i*(x_i - y_i)^2 (non-squared w_user_i here)
        # Also, pit (point-in-time) properties are still replicated across all time points
        # but will have the same std as non-replicated set
        self._logger.info("Normalization/weighting ...")
        self.pit_properties = []
        self.ts_properties = []
        self.full_scaling = {} # keep these in case we need to deal with seed regions outside search region
        self.past_seeds = {} # invalidate seed region cache since scaling might be about to change
        for c in self.data.columns:
            stdev = self.data[c].std()
            # always report true stds - might want to include them in properties.py as 'norm' on later runs
            self._logger.info("Data standard deviation for {} is {}".format(c,stdev))
            self.full_scaling[c] = np.sqrt(self.metric_weights[c]) / self.metric_properties[c]['properties'].get("norm",stdev)
            self.data[c] *= self.full_scaling[c]

            # will use to split dataset into time series and pit parts
            col_type = self.metric_properties[c]["properties"]["type"]
            if col_type == "pit":
                self.pit_properties.append(c)
            else:
                self.ts_properties.append(c)

        self.n_pit = len(self.pit_properties)
        self.n_ts = len(self.ts_properties)
        return

    def _reformat(self):
        ########################## Reformat data into single row per region.##############
        # First n_prop columns are means (for pit it will be values themselves, for ts - yearly means),
        # followed by n_ts*n_ts giving covar matrix of time series properties
        # (matrix is symmetric, so wasting space but makes life easier)
        self._logger.info("Computing means/covars and reformatting ...")
        # do not dropna when computing  means - probably get better averages this way
        data_means = self.data.groupby(level='region').mean()
        #ts_means = self.data[self.ts_properties].groupby(level='region').mean()
        if self.n_ts > 0:
            # dropna keeps only valid points for given region if drop_mode=fully_missing (no-op f drop_mode=any_missing)
            covars = self.data.dropna()[self.ts_properties].groupby(level='region').cov().unstack(level=1)
            self.data = np.concatenate([data_means.values,covars.values], axis=1)
        else:
            self.data = data_means.values
        return



    def _merge_into_prop_data(self, from_data, from_regions):
        # map region to its location in the original list (and therefore from_data array)
        from_map = dict(zip(from_regions,range(len(from_regions))))
        av_regions = sorted(from_regions) #self.needed_regions is already sorted
        i=0
        j=0
        while i<len(av_regions) and j<len(self.needed_regions):
            r_from = av_regions[i]
            r_to = self.needed_regions[j]
            if r_from == r_to:
                # found the same region in both lists
                # for prop_data use index into self.needed_regions directly
                # for from_data go through the map (index is into sorted version)
                k = from_map[r_from]
                self.prop_data[j*self.t_int_per_year:(j+1)*self.t_int_per_year] = \
                     from_data[k*self.t_int_per_year:(k+1)*self.t_int_per_year]
                # advance both 'to' and 'from'
                i += 1
                j += 1
            elif r_from > r_to:
                # advance 'to'
                j +=1
            else:
                # advance 'from'
                i +=1
        # reached the end of either 'from' or 'to' lists, but in either case
        # nothing else to do - remaining regions are either not needed or not available
        return
        # The code above is equivalent to one below but index() is linear time in list length
        # so this runs in len(from_regions)*len(self.needed_regions) time
        # which can be quite slow for long lits
        #for (i,r) in enumerate(from_regions):
        #    try:
        #        idx = self.needed_regions.index(r)
        #        self.prop_data[idx*self.t_int_per_year:(idx+1)*self.t_int_per_year] = from_data[i*self.t_int_per_year:(i+1)*self.t_int_per_year]
        #    except:
        #        # we do not need this region
        #        continue

    def _load_property_from_file(self, prop_name, path_prefix):
        # Reads data from region_name.prop_name.csv file
        # Three columns expected: start_date,end_date and value. To maintain uniform format,
        # his is expected even for static properties, but for these we
        # simply take value from the first line, the rest of the file is ignored

        full_path = path_prefix + '.'+prop_name+'.csv'
        # will return array of long-term averages for each time interval within a year, even for static
        result = np.zeros(self.t_int_per_year)
        try:
            raw_data = pd.read_csv(full_path, parse_dates=[0,1], infer_datetime_format=True)
        except Exception as e:
            self._logger.error("Can not get data from file {}: {}".format(full_path, str(e)))
            return None, [] # empty list signals no data

        if prop_name in self.ts_properties:
            # this is re-implementation of the code in _fill_block
            # where it was done for a single value
            nt = raw_data.shape[0]
            fractions = np.zeros((nt,self.t_int_per_year))
            raw_data['start_doy'] = raw_data['start_date'].dt.dayofyear
            raw_data['end_doy'] = raw_data['end_date'].dt.dayofyear
            start_f = ((raw_data['start_doy']-1)/self.t_int_days).clip(upper=self.t_int_per_year-1e-9)
            end_f = (raw_data['end_doy']/self.t_int_days).clip(upper=self.t_int_per_year-1e-9)
            (start_interval,start_fraction) = np.divmod(start_f.values,1)
            (end_interval,end_fraction) = np.divmod(end_f.values,1)
            start_interval = start_interval.astype(int)
            end_interval = end_interval.astype(int)
            start_fraction = 1-start_fraction
            fractions[range(nt),start_interval] += start_fraction
            fractions[range(nt),end_interval] += end_fraction
            # adjust lines contributing to a single interval
            fractions = np.where(np.tile(start_interval==end_interval,(self.t_int_per_year,1)).T & (fractions>0),
                                 fractions-1, fractions)

            # fills in ones strictly between start_interval and end_intervals (not ends)
            mask = np.tile(np.array(range(self.t_int_per_year)),[nt,1])
            mask = (mask>start_interval.reshape(-1,1)) & (mask<end_interval.reshape(-1,1))
            fractions[mask] = 1
            # We remove data points crossing year boundary (revisit later?)
            # and average raw data according to contributions of each line to each interval
            # result will have nan's for intervals with no coverage (division by zero warnings expected in this case)
            counters = fractions[start_interval<=end_interval,:].sum(axis=0)
            result = (raw_data['value'].values.reshape(-1,1)*fractions)[start_interval<=end_interval,:].sum(axis=0) / counters
        else:
            result += raw_data['value'].iloc[0]
        return result, [-1] # return any non-empty list (contains actual region list for database read)


    def _load_property_for_regions(self, prop_name, regions_list, track_na=True, depth=0):
        is_ts = self.metric_properties[prop_name]["properties"]["type"] == "timeseries"
        query = self.metric_properties[prop_name]["api_data"]
        queries = []
        n_reg = len(regions_list)
        n_queries =  n_reg // REGIONS_PER_QUERY + (n_reg % REGIONS_PER_QUERY != 0)

        # allocate full size, but is being filled in in order of quert completion
        valid_data = np.zeros(n_reg*self.t_int_per_year)
        data_counters = np.zeros(n_reg*self.t_int_per_year)

        # every region ends up on either valid_regions or to_retry
        valid_regions = []
        self.to_retry = []
        processed_q = []
        self.finished = True # we always hope that current run is the last

        for q in range(n_queries):
            copy_of_metric = dict(query)
            # ok for last index to exeed list length min(n_reg, (q+1)*REGIONS_PER_QUERY) is used
            copy_of_metric["region_id"] = regions_list[q*REGIONS_PER_QUERY:(q+1)*REGIONS_PER_QUERY]
            queries.append(copy_of_metric)

        #********************
        # function to process API responce to given query
        # self.data should already have space allocated for all data
        def map_response(idx, _, response, *args):
            r_idx = regions_list[idx*REGIONS_PER_QUERY:(idx+1)*REGIONS_PER_QUERY]
            # if entire response is invalid, assume this query should probably be rerun
            # => don't add to processed list. Should pay attention to messages if None is legitimate (will repeat on resume)
            if (response is None) or (type(response) is BatchError) or (len(response) == 0):
                self.finished = False
                self._logger.info("Received None or BatchError as response to query {} for regions {} to {}".
                                  format(idx,r_idx[0],r_idx[-1]))
                self.to_retry += r_idx
                if type(response) is BatchError:
                    self._logger.info("received BatchError(response, retry_count, url, params): {}".format(response))
                return
            for r in r_idx:
                # try/exept to avoid crash on invalid data, just pass None to _fill_block to deal with it properly
                try:
                    resp = [{key:item[key] for key in ['start_date', 'end_date','value']} for item in response if item['region_id']==r]
                except:
                    resp = None
                if self._fill_block(resp, prop_name, len(valid_regions)*self.t_int_per_year, valid_data, data_counters, track_na):
                    valid_regions.append(r)
                else:
                    self._logger.debug("Did not get any {} for region {}".format(prop_name,r))
                    self.to_retry.append(r)
                load_bar.update()
                processed_q.append(idx)
            return

        #***********************

        self._logger.info("Getting data series for {} regions in {} queries for property {}".
                          format(n_reg, n_queries, prop_name))
        load_bar = tqdm(total=n_reg)
        # Sending out all queries
        self.client.batch_async_get_data_points(queries, map_result=map_response)
        load_bar.close()
        data_counters[data_counters == 0] = np.nan # avoid division by zero warning, where no data
        valid_data = valid_data/data_counters
        #valid_data = valid_data[np.isfinite(valid_data)]
        # cut off the actually filled-in part (valid regions only)
        valid_data=valid_data[:len(valid_regions)*self.t_int_per_year]

        # have a list of (un)processed queries and a bunch of random regins we did not received any data for
        if not self.finished and depth < MAX_RETRY_FAILED_REGIONS:
            self._logger.info("Retry {} failed regions at depth {}...".format(len(self.to_retry), depth))
            (extra_data,extra_regions) = self._load_property_for_regions(prop_name, self.to_retry, track_na=False, depth=depth+1)
            valid_data = np.concatenate([valid_data,extra_data], axis=0)
            valid_regions += extra_regions

        return (valid_data,valid_regions)

    ##############################
    # fills corresponding block of data/data_counters (starting from idx) from API response
    # returns True if actual data were received, False for empty/None response
    def _fill_block(self, response, prop_name, idx, data, data_counters, track_na=True):
        res = True
        #print("fb ", data.shape, idx)
        if response is None or len(response) == 0 or (len(response) == 1 and response[0] == {}):
            # no data on this region/property. let's fill it with nan
            data[idx:idx+self.t_int_per_year] = np.nan
            res = False
        else:
            if self.metric_properties[prop_name]["properties"]["type"] == "timeseries":
                for datapoint in response:
                    v = datapoint["value"]

                    # No action on N/A
                    if v is not None:
                        # Dates are recieved as starting from YYYY-MM-DD
                        # (for example 'start_date': '2017-01-01T00:00:00.000Z')
                        # ugly date conversion, but this is inner loop, strptime is too slow here
                        # TODO: python 3.7 has fromisoformat which is supposedly much faster than strptime
                        y = int(datapoint["start_date"][:4])
                        m = int(datapoint["start_date"][5:7])
                        d = int(datapoint["start_date"][8:10])
                        start_doy = date(y,m,d).timetuple()[-2]
                        #int(datetime.datetime.strptime(datapoint["start_date"][:10], '%Y-%m-%d').strftime('%j'))
                        y = int(datapoint["end_date"][:4])
                        m = int(datapoint["end_date"][5:7])
                        d = int(datapoint["end_date"][8:10])
                        end_doy = date(y,m,d).timetuple()[-2]
                        #int(datetime.datetime.strptime(datapoint["end_date"][:10], '%Y-%m-%d').strftime('%j'))

                        # scale into [0,t_int_per_year]
                        # intervals are INCLUSIVE of days stating their ends
                        # for 30.4-day interval, start_f=29/30.4 for day 30, 30/30.4 for day 31.
                        # but end_f = 30/30.4 for day 30. So single day 30 maps to [29,30]/30.4
                        # (fully inside interval 0) but 31 is partially split between intervals 0 and 1
                        # -1e-9 makes sure end boundary is strictly inside interval (also handles leap years)
                        start_f = min(self.t_int_per_year-1e-9, (start_doy-1)/self.t_int_days)
                        end_f = min(self.t_int_per_year-1e-9, end_doy/self.t_int_days)
                        (start_interval,start_fraction) = divmod(start_f,1)
                        (end_interval,end_fraction) = divmod(end_f,1)
                        start_interval = int(start_interval)
                        end_interval = int(end_interval)
                        start_fraction = 1-start_fraction
                        fractions = np.ones(end_interval-start_interval+1)

                        if end_interval==start_interval:
                            # data point is fully inside an interval - only one interval affected
                            fractions[0]=start_fraction+end_fraction-1
                        elif end_interval>start_interval:
                            # end interval is later in the year than start (no crossing of year boundary)
                            fractions[0]=start_fraction
                            fractions[-1]=end_fraction
                        else:
                            # crossing year boundary - let's just ignore this point
                            continue
                        data_counters[idx+start_interval:idx+end_interval+1] += fractions
                        data[idx+start_interval:idx+end_interval+1] += v*fractions
            elif self.metric_properties[prop_name]["properties"]["type"] == "pit":
                # for 'point in time' just add the same value across all months
                v =  response[0]["value"]
                # On None can assign np.nan immediately,
                # but let's keep uniformity with timeseries and take no action
                if v is None:
                    res = False
                else:
                    data[idx:idx+self.t_int_per_year] = v
                    data_counters[idx:idx+self.t_int_per_year] += 1
        return res


    def _get_distances(self, idx1, idx2, full_dist):
        """ Returns the distances in each property for the two given rows
            Distances are in the final space, i.e. incorporate all aplied normalization/weighting
        """
        means1 = self.data[idx1,:self.n_pit+self.n_ts]
        means2 = self.data[idx2,:self.n_pit+self.n_ts]
        s_dist = np.sqrt(max(0,full_dist**2 - np.sum((means1-means2)**2)))
        distances = dict([('total',full_dist), ('covar',s_dist)]+[(p,np.abs(means1[i]-means2[i])) for (i,p) in enumerate(self.needed_properties)])
        return distances

    def _get_distances_means(self, means1, idx2, full_dist):
        """ Same as above but takes means of seed region as argument instead of its index into data array
            (useful for seed regions outside search region)
        """
        means2 = self.data[idx2,:self.n_pit+self.n_ts]
        s_dist = np.sqrt(max(0,full_dist**2 - np.sum((means1-means2)**2)))
        distances = dict([('total',full_dist), ('covar',s_dist)]+[(p,np.abs(means1[i]-means2[i])) for (i,p) in enumerate(self.needed_properties)])
        return distances

    def similar_to(self, region_id, compare_to=0, number_of_regions=10, requested_level=None, detailed_distance=False):
        """
        Attempt to look up the given name and find similar regions.
        :param region_id: a Gro region id representing the reference region you want to find similar regions to.
            If a string is given, it is interpreted as a prefix path for a set of .csv per-property files for a seed region
        :param compare_to: the root region_id of the regions to compare to (default 0 which is World)
        :param number_of_regions: number of most similar matches to return
        :param requested_level: level of returned regions (3-country,4-province,5-district)
        :return: a generator of the most similar regions as a list in the form
        {'#': 0, 'id': 123, 'name': "abc", 'dist': 1.23, 'parent': (12,"def",,)}
        """
        if self.search_region != compare_to:
            self.build(compare_to)

        # To change search region, user should construct new SR object
        # but we want to allow searches with seed region outside serach region,
        # say, seed region in South America but search region is Africa
        #
        # For simplicity, in such cases always download data for the seed region from the system directly (ignore cache)
        if region_id not in self.available_regions:
            self._logger.info("seed region {} is outside search region".format(region_id))
            x = self.past_seeds.get(region_id)
            if x is None:
                seed_region_means = []
                seed_region_ts = pd.DataFrame(columns=self.ts_properties)
                for prop_name in self.needed_properties:
                    # if string is given load from a file
                    if isinstance(region_id, str):
                        valid_data, valid_regions = self._load_property_from_file(prop_name, region_id)
                    else:
                        valid_data, valid_regions = self._load_property_for_regions(prop_name, [region_id])
                    # seed region should have all required data - bail out if anything missing
                    if not valid_regions:
                        self._logger.error("could not get {} data for requested seed region {}".format(prop_name, region_id))
                        return
                    valid_data *= self.full_scaling[prop_name]
                    # neen mean regardless - for pit will be the same as (duplicated) value
                    seed_region_means.append(valid_data.mean())
                    if prop_name in self.ts_properties:
                        seed_region_ts[prop_name] = valid_data
                # once all properties are collected, generate full data row for seed region
                if self.n_ts > 0:
                    seed_region_covar = seed_region_ts[self.ts_properties].cov().unstack()
                    seed_region_data = np.concatenate([np.array(seed_region_means),seed_region_covar.values])
                else:
                    seed_region_data = np.array(seed_region_means)
                x = seed_region_data.reshape(1,-1)
                self.past_seeds[region_id] = x
            else:
                self._logger.info("but was found in seed region cache")
            #print(x[0])
        else:
            # seed region present in the data => just get corresponding row

            # called when self.data is simple np.array
            # BallTree expects array of points but we always search neighbors of just one => reshape
            x = self.data[self.region_index[region_id]].reshape(1, -1)

        # distances and regions (as indexes into self.data)
        # Always use single lookup region - use [0] of the result
        max_num_regions = self.num_regions
        if requested_level:
            ball = self.balls_on_level.get(requested_level)
            assert ball, "No region data on requested level {}".format(requested_level)
            idx_to_region = self.regions_on_level[requested_level]
            max_num_regions = len(idx_to_region)
        else:
            ball = self.ball
            idx_to_region = self.available_regions
        sim_dists, neighbour_idxs = ball.query(x, k = min(number_of_regions, max_num_regions))
        sim_regions = [idx_to_region[i] for i in neighbour_idxs[0]]
        sim_dists = sim_dists[0]
        if detailed_distance:
            #r_idx = self.region_index[region_id]
            #sim_dists = [self._get_distances(r_idx, self.region_index[r], sim_dists[i])
            #             for (i,r) in enumerate(sim_regions)]
            sim_dists = [self._get_distances_means(x[0][:self.n_pit+self.n_ts], self.region_index[r], sim_dists[i])
                         for (i,r) in enumerate(sim_regions)]
        self._logger.info("Found {} regions most similar to '{}'.".format(len(sim_regions), region_id))

        for ranking, sr_id in enumerate(sim_regions):
            info = self.region_info[sr_id]

            # Choose parent one level up from this region
            parent_info = {"name": "", "id": ""}
            for r in info['belongsTo']:
                # in case there is no parent at correct level we will just take the last
                try:
                    parent_info = self.region_info[r]
                    if parent_info['level'] == region_level-1:
                        break
                except:
                    continue

            # try to find grandparent in region info, just take the first id parent belongs to
            # this works ok for grandparents which are at least countries if parent on needed level was found
            try:
                gp_id = self.region_info[parent_info['id']]['belongsTo'][0]
                gp_info = self.region_info[gp_id]
            except:
                gp_info = {"name": "", "id": ""}

            data_point = {"#": ranking,
                          "id": sr_id,
                          "name": info['name'],
                          "dist": sim_dists[ranking],
                          "parent": (parent_info["id"], parent_info["name"], gp_info["id"], gp_info["name"])}
            yield data_point
