"""Base module for making API requests.

Client, GroClient, CropModel, and BatchClient all build on top of endpoints
exposed in this module. Helper functions or shims or derivative functionality
should appear in the client classes rather than here.
"""

from builtins import map
from builtins import str
from api.client import cfg
import json
import logging
import requests
import time
from datetime import datetime
try:
    # functools are native in Python 3.2.3+
    from functools import lru_cache as memoize
except ImportError:
    from backports.functools_lru_cache import lru_cache as memoize

REGION_LEVELS = {
    'world': 1,
    'continent': 2,
    'country': 3,
    'province': 4,  # Equivalent to state in the United States
    'district': 5,  # Equivalent to county in the United States
    'city': 6,
    'market': 7,
    'other': 8,
    'coordinate': 9
}

GRAPH_TYPES = ['metric', 'item', 'region']
EXPANDABLE_TYPES = GRAPH_TYPES + ['partner_region']
SERIES_TYPES = EXPANDABLE_TYPES + ['frequency', 'source']
LOOKUP_TYPES = SERIES_TYPES + ['unit']


"""
------------------------
| Parameter formatters |
------------------------
"""


def as_singular_id(types_singular_list):
    """Convert a list of type_singular format types into type_singular_id format.

    Parameters
    ----------
    types_singular_list : list of strings
        GRAPH_TYPES, EXPANDABLE_TYPES, SERIES_TYPES, or LOOKUP_TYPES

    >>> as_singular_id(['metric', 'frequency', 'source', 'partner_region'])
    ['metric_id', 'frequency_id', 'source_id', 'partner_region_id']

    """
    return [type_singular+'_id' for type_singular in types_singular_list]


def as_type_plural(types_singular_list):
    """Convert a list of type_singular format types into type_singular_id format.

    Parameters
    ----------
    types_singular_list : list of strings
        GRAPH_TYPES, EXPANDABLE_TYPES, SERIES_TYPES, or LOOKUP_TYPES

    >>> as_type_plural(['metric', 'frequency', 'source', 'partner_region'])
    ['metrics', 'frequencies', 'sources', 'partner_regions']

    """
    return [type_singular[:-1] + 'ies' if type_singular[-1] == 'y' else type_singular+'s'
            for type_singular in types_singular_list]


@memoize(maxsize=None)
def snake_to_camel(term):
    """Convert a string from snake_case to camelCase.

    >>> snake_to_camel('hello_world')
    'helloWorld'

    Parameters
    ----------
    term : string

    Returns
    -------
    string

    """
    camel = term.split('_')
    return ''.join(camel[:1] + list([x[0].upper()+x[1:] for x in camel[1:]]))


def get_params(allowed_inputs, selection):
    """Construct http request params from dict of entity selections.

    For use with get_data_points().

    >>> get_data_call_params(['metric_id', 'item_id', 'start_date'],
    ...                      { 'metric_id': 123, 'start_date': '2012-01-01', 'unit_id': 14 }) == {
    ...     'metricId': 123, 'startDate': '2012-01-01' }
    True

    Returns
    -------
    dict
        selections with valid keys converted to camelcase and invalid ones filtered out

    """
    params = {}
    for key, value in selection.items():
        if key in allowed_inputs:
            params[snake_to_camel(key)] = value
    return params


def assert_types(key, value):
    """Raise an AssertionError if key is not the right type.

    Parameters
    ----------
    key : string
    value : any

    Raises
    ------
    AssertionError

    """
    if key == 'type_singular':
        assert value in LOOKUP_TYPES
    elif key == 'type_plural':
        assert value in as_type_plural(LOOKUP_TYPES)
    elif key in LOOKUP_TYPES or key in as_singular_id(LOOKUP_TYPES):
        assert isinstance(value, int), "{} must be an integer".format(key)
    elif key in as_type_plural(LOOKUP_TYPES):
        assert isinstance(value, list) and len(value) > 0 and isinstance(value[0], int),\
            "{} must be a list of integers".format(key)
    elif key in ['start_date', 'end_date', 'at_time']:
        assert (isinstance(value, str) and
                value == datetime.strptime(value, "%Y-%m-%d").strftime('%Y-%m-%d')),\
               "{} must be a YYYY-MM-DD formatted date string".format(key)
    elif key in ['insert_null']:
        assert isinstance(value, bool), "{} must be a boolean".format(key)
    else:
        raise AssertionError('Unknown query parameter')


def get_default_logger():
    """Get a logging object using the default log level set in cfg.

    https://docs.python.org/3/library/logging.html

    Returns
    -------
    logger : logging.Logger

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(cfg.DEFAULT_LOG_LEVEL)
    if not logger.handlers:
        stderr_handler = logging.StreamHandler()
        logger.addHandler(stderr_handler)
    return logger


def get_access_token(api_host, user_email, user_password, logger=None):
    """Request an access token.

    Parameters
    ----------
    api_host : string
        The API host's url, excluding 'https://'
        ex. 'api.gro-intelligence.com'
    user_email : string
        Email address associated with user's Gro account
    user_password : string
        Password for user's Gro account
    logger : logging.Logger
        Alternative logger object if wanting to use a non-default one.
        Otherwise get_default_logger() will be used.

    Returns
    -------
    accessToken : string

    """
    retry_count = 0
    if not logger:
        logger = get_default_logger()
    while retry_count < cfg.MAX_RETRIES:
        get_api_token = requests.post('https://' + api_host + '/api-token',
                                      data={'email': user_email,
                                            'password': user_password})
        if get_api_token.status_code == 200:
            logger.debug('Authentication succeeded in get_access_token')
            return get_api_token.json()['data']['accessToken']
        else:
            logger.warning('Error in get_access_token: {}'.format(
                get_api_token))
        retry_count += 1
    raise Exception('Giving up on get_access_token after {0} tries.'.format(
        retry_count))


def redirect(old_params, migration):
    """Update query parameters to follow a redirection response from the API.

    >>> redirect(
    ...     {'metricId': 14, 'sourceId': 2, 'itemId': 145},
    ...     {'old_metric_id': 14, 'new_metric_id': 15, 'source_id': 2}
    ... ) == {'metricId': 15, 'sourceId': 2, 'itemId': 145}
    True

    Parameters
    ----------
    old_params : dict
        The original parameters provided to the API request
    migration : dict
        The body of the 301 response indicating which of the inputs have been
        migrated and what values they have been migrated to

    Returns
    -------
    new_params : dict
        The mutated params object with values replaced according to the
        redirection instructions provided by the API

    """
    new_params = old_params.copy()
    for migration_key in migration:
        split_mig_key = migration_key.split('_')
        if split_mig_key[0] == 'new':
            param_key = snake_to_camel('_'.join([split_mig_key[1], 'id']))
            new_params[param_key] = migration[migration_key]
    return new_params


def get_data(url, headers, params=None, logger=None):
    """General 'make api request' function.

    Assigns headers and builds in retries and logging.

    Parameters
    ----------
    url : string
    headers : dict
    params : dict
    logger : logging.Logger

    Returns
    -------
    data : list or dict

    """
    base_log_record = dict(route=url, params=params)
    retry_count = 0
    if not logger:
        logger = get_default_logger()
        logger.debug(url)
        logger.debug(params)
    while retry_count < cfg.MAX_RETRIES:
        start_time = time.time()
        data = requests.get(url, params=params, headers=headers, timeout=None)
        elapsed_time = time.time() - start_time
        log_record = dict(base_log_record)
        log_record['elapsed_time_in_ms'] = 1000 * elapsed_time
        log_record['retry_count'] = retry_count
        log_record['status_code'] = data.status_code
        if data.status_code == 200:
            logger.debug('OK', extra=log_record)
            return data
        if data.status_code == 204:
            logger.warning('No Content', extra=log_record)
            return data
        retry_count += 1
        log_record['tag'] = 'failed_gro_api_request'
        if retry_count < cfg.MAX_RETRIES:
            logger.warning(data.text, extra=log_record)
        if data.status_code == 429:
            time.sleep(2 ** retry_count)  # Exponential backoff before retrying
        elif data.status_code == 301:
            new_params = redirect(params, data.json()['data'][0])
            logger.warning('Redirecting {} to {}'.format(params, new_params), extra=log_record)
            params = new_params
        elif data.status_code in [404, 401, 500]:
            break
        else:
            logger.error('{}'.format(data), extra=log_record)
    raise Exception('Giving up on {} after {} tries: {}.'.format(
        url, retry_count, data))


@memoize(maxsize=None)
def get_available(access_token, api_host, entity_type):
    assert_types('type_plural', entity_type)
    url = '/'.join(['https:', '', api_host, 'v2', entity_type])
    headers = {'authorization': 'Bearer ' + access_token}
    resp = get_data(url, headers)
    return resp.json()['data']


def list_available(access_token, api_host, selected_entities):
    params = get_params(SERIES_TYPES, selected_entities)
    url = '/'.join(['https:', '', api_host, 'v2/entities/list'])
    headers = {'authorization': 'Bearer ' + access_token}
    resp = get_data(url, headers, params)
    try:
        return resp.json()['data']
    except KeyError:
        raise Exception(resp.text)


@memoize(maxsize=None)
def lookup(access_token, api_host, entity_type, entity_id):
    assert_types('type_singular', entity_type)
    assert_types('id', entity_id)
    url = '/'.join(['https:', '', api_host, 'v2', entity_type, str(entity_id)])
    headers = {'authorization': 'Bearer ' + access_token}
    resp = get_data(url, headers)
    try:
        return resp.json()['data']
    except KeyError:
        raise Exception(resp.text)


def get_data_series(access_token, api_host, **selection):
    logger = get_default_logger()
    url = '/'.join(['https:', '', api_host, 'v2/data_series/list'])
    headers = {'authorization': 'Bearer ' + access_token}
    params = get_params(SERIES_TYPES)
    resp = get_data(url, headers, params)
    try:
        response = resp.json()['data']
        if any((series.get('metadata', {}).get('includes_historical_region', False))
               for series in response):
            logger.warning('Some of the regions in your data call are historical, with boundaries '
                           'that may be outdated. The regions may have overlapping values with '
                           'current regions')
        return response
    except KeyError:
        raise Exception(resp.text)


def get_source_ranking(access_token, api_host, series):
    """Given a series, return a list of ranked sources.

    Parameters
    ----------
    access_token : string
    api_host : string
    series: dict
        Series to calculate source raking for.

    Returns
    -------
    list of integers
        Source ids that match the series parameters, sorted by rank

    """
    valid_inputs = as_type_plural(SERIES_TYPES)+['start_date', 'end_date']
    url = '/'.join(['https:', '', api_host, 'v2/available/sources'])
    headers = {'authorization': 'Bearer ' + access_token}
    return get_data(url, headers, params).json()


def rank_series_by_source(access_token, api_host, series_list):
    # We sort the internal tuple representations of the dictionaries because
    # otherwise when we call set() we end up with duplicates if iteritems()
    # returns a different order for the same dictionary. See test case.
    selections_sorted = set(tuple(sorted(
        [k_v for k_v in iter(list(single_series.items()))
         if k_v[0] not in ('source_id', 'source_name')],
        key=lambda x: x[0])) for single_series in series_list)

    for series in map(dict, selections_sorted):
        try:
            source_ids = get_source_ranking(access_token, api_host, series)
        except ValueError:
            continue  # empty response
        for source_id in source_ids:
            # Make a copy to avoid passing the same reference each time.
            series_with_source = dict(series)
            series_with_source['source_id'] = source_id
            yield series_with_source


def format_list_of_series(series_list):
    """Convert list_of_series format from API back into the familiar single_series output format.

    Parameters
    ----------
    series_list : list of dicts
        "list_of_series" format. Each dict has "series" and "data" attributes. "series" is a dict,
        and "data" is a list of 3-5 element lists in the order
        [start_date, end_date, value, reporting_date, metadata]. Reporting_date or metadata may be
        excluded or None if there is none available.

    Returns
    -------
    list of dicts
        "single_series" format. Each dict is a different time series point, and every one of them
        has the attributes of the series the data point belongs to.

    >>> format_list_of_series([{ 'series': {}, 'data': [['2001-01-01', '2001-12-31', 123]] }]) == [
    ...   { 'start_date': '2001-01-01', 'end_date': '2001-12-31', 'value': 123,
    ...     'reporting_date': None, 'metric_id': None, 'item_id': None, 'region_id': None,
    ...     'partner_region_id': 0, 'frequency_id': None, 'source_id': None, 'unit_id': None,
    ...     'belongs_to': {} } ]
    True

    """
    if(isinstance(series_list, list)):
        output = []
        for series in series_list:
            if(isinstance(series, dict) and isinstance(series.get('data', []), list)):
                for point in series.get('data', []):
                    single_series_point = {
                        'start_date': point[0],
                        'end_date': point[1],
                        'value': point[2],
                        'reporting_date': point[3] if len(point) > 3 else None,
                        'metric_id': series['series'].get('metricId', None),
                        'item_id': series['series'].get('itemId', None),
                        'region_id': series['series'].get('regionId', None),
                        'partner_region_id': series['series'].get('partnerRegionId', 0),
                        'frequency_id': series['series'].get('frequencyId', None),
                        'source_id': series['series'].get('sourceId', None),
                        'unit_id': series['series'].get('unitId', None),
                        'belongs_to': series['series'].get('belongsTo', {})
                    }
                    output.append(single_series_point)
        return output
    # If the output is an error or None or something else that's not a list, just propagate
    return series_list


def get_data_points(access_token, api_host, **selection):
    headers = {'authorization': 'Bearer ' + access_token}
    url = '/'.join(['https:', '', api_host, 'v2/data'])
    valid_inputs = SERIES_TYPES + ['start_date', 'end_date', 'at_time', 'insert_null']
    params = get_params(valid_inputs, selection)
    resp = get_data(url, headers, params)
    return format_list_of_series(resp.json())


@memoize(maxsize=None)
def universal_search(access_token, api_host, search_terms):
    """Search across all entity types for the given terms.

    Parameters
    ----------
    access_token : string
    api_host : string
    search_terms : string

    Returns
    -------
    list of [id, entity_type] pairs

        Example::

            [[5604, 'item'], [10204, 'item'], [410032, 'metric'], ....]

    """
    url_pieces = ['https:', '', api_host, 'v2/search']
    url = '/'.join(url_pieces)
    headers = {'authorization': 'Bearer ' + access_token}
    resp = get_data(url, headers, {'q': search_terms})
    return resp.json()


@memoize(maxsize=None)
def search(access_token, api_host, entity_type, search_terms):
    assert_types('type_plural', entity_type)
    url = '/'.join(['https:', '', api_host, 'v2/search', entity_type])
    headers = {'authorization': 'Bearer ' + access_token}
    resp = get_data(url, headers, {'q': search_terms})
    return resp.json()


def search_and_lookup(access_token, api_host, entity_type, search_terms, num_results=10):
    assert_types('type_plural', entity_type)
    search_results = search(access_token, api_host, entity_type, search_terms)
    for result in search_results[:num_results]:
        yield lookup(access_token, api_host, entity_type, result['id'])


def lookup_belongs(access_token, api_host, entity_type, entity_id):
    assert_types('type_plural', entity_type)
    assert_types('id', entity_id)
    url = '/'.join(['https:', '', api_host, 'v2', entity_type, 'belongs-to'])
    params = {'ids': str(entity_id)}
    headers = {'authorization': 'Bearer ' + access_token}
    parents = get_data(url, headers, params).json().get('data').get(str(entity_id), [])
    for parent_entity_id in parents:
        yield lookup(access_token, api_host, entity_type, parent_entity_id)


def get_geo_centre(access_token, api_host, region_id):
    assert_types('id', region_id)
    url = '/'.join(['https:', '', api_host, 'v2/geocentres?regionIds=' +
                    str(region_id)])
    headers = {'authorization': 'Bearer ' + access_token}
    resp = get_data(url, headers)
    return resp.json()['data']


@memoize(maxsize=None)
def get_geojson(access_token, api_host, region_id):
    assert_types('id', region_id)
    url = '/'.join(['https:', '', api_host, 'v2/geocentres?includeGeojson=True&regionIds=' +
                    str(region_id)])
    headers = {'authorization': 'Bearer ' + access_token}
    resp = get_data(url, headers)
    for region in resp.json()['data']:
        return json.loads(region['geojson'])
    return None


def get_descendant_regions(access_token, api_host, region_id,
                           descendant_level=False, include_historical=True):
    assert_types('id', region_id)
    descendants = []
    region = lookup(access_token, api_host, 'regions', region_id)
    for member_id in region['contains']:
        member = lookup(access_token, api_host, 'regions', member_id)
        if (not include_historical and member['historical']):
            continue
        if not descendant_level or descendant_level == member['level']:
            descendants.append(member)
        if not descendant_level or member['level'] < descendant_level:
            descendants += get_descendant_regions(
                access_token, api_host, member_id, descendant_level, include_historical)
    return descendants


if __name__ == '__main__':
    # To run doctests:
    # $ python lib.py -v
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
