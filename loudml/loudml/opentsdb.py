"""
OpenTSDB module for Loud ML
"""
from loudml.bucket import Bucket
import requests
import pandas

# json imports for tsdb

try:
    # Use ujson if available.
    import ujson as json
except Exception:
    import json

from voluptuous import (
    Required,
    Optional,
    All,
    Length,
    Boolean,
)

from . import (
    errors,
    schemas,
)
from loudml.misc import (
    escape_quotes,
    escape_doublequotes,
    make_ts,
    parse_addr,
    str_to_ts,
    ts_to_str,
)

class OpenTSDB(Bucket):
    """
    OpenTSDB Bucket
    """

    SCHEMA = Bucket.SCHEMA.extend({
        Required('addr'): str,
    })

    def __init__(self, cfg):
        cfg['type'] = 'opentsdb'
        super().__init__(cfg)

    def get_times_data(
        self,
        bucket_interval,
        features,
        from_date=None,
        to_date=None,
    ):
        nb_features = len(features)

        queries = self._build_times_queries(
            bucket_interval, features, from_date, to_date)
        results = self._run_queries(queries)

        if not isinstance(results, list):
            results = [results]

        buckets = []
        # Merge results
        for i, result in enumerate(results):
            feature = features[i]

            for j, point in enumerate(result.get_points()):
                agg_val = point.get(feature.name)
                timeval = point['time']

                if j < len(buckets):
                    bucket = buckets[j]
                else:
                    bucket = {
                        'time': timeval,
                        'mod': int(str_to_ts(timeval)) % bucket_interval,
                        'values': {},
                    }
                    buckets.append(bucket)

                bucket['values'][feature.name] = agg_val

        # Build final result
        t0 = None
        result = []

        for bucket in buckets:
            X = np.full(nb_features, np.nan, dtype=float)
            timeval = bucket['time']
            ts = str_to_ts(timeval)

            for i, feature in enumerate(features):
                agg_val = bucket['values'].get(feature.name)
                if agg_val is None:
                    logging.info(
                        "missing data: field '%s', metric '%s', bucket: %s",
                        feature.field, feature.metric, timeval,
                    )
                else:
                    X[i] = agg_val

            if t0 is None:
                t0 = ts

            result.append(((ts - t0) / 1000, X, timeval))

        return result


###### old code
class OpenTSDBResponseSerie(object):
    """
        A single OpenTSDB response serie i.e 1 element of the response
        array.
        Params:
            **kwargs : OpenTSDB response serie data
    """
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    @property
    def id(self):
        """
            id for serie
            Returns:
                metric{sorted=tag,key=value}
        """
        if len(self.tags.keys()) > 0:
            tags = ",".join(["%s=%s" %
                (k, self.tags[k]) for k in sorted(self.tags.keys())])
            return "%s{%s}" % (self.metric, tags)
        else:
            return self.metric


    def alias(self, functOrStr):
        """
            User specified alias using lambda functions and string formatting using
            metadata provided by opentsdb.
            This function fails silently.
            Params:
                functOrStr :    lambda function or python string format. When using lambda
                                functions,  they must begin with '!' e.g. !lambda x: x....
            Return:
                Formatted alias on success and id or failure.
        """
        flatData = self.__flattenedMetadata()
        # Normalized alias
        _alias = ""
        if functOrStr.startswith("!"):
            try:
                _alias = eval(functOrStr[1:])(flatData)
            except Exception as e:
                pass
        else:
            try:
                _alias = functOrStr % (flatData)
            except Exception as e:
                pass

        if _alias == "":
            return self.id

        return _alias


    def __flattenedMetadata(self):
        """
            Flattens all metadata which is used for normalization
        """
        return dict([("metric", self.metric)] +
            [("tags.%s" % (k), v) for k, v in self.tags.items()])

    def datapoints(self, convertTime=False):
        """
            Converts datapoints
            Params:
                convertTime : Whether to convert epoch to pandas datetime
            Return:
                Array of tuples (time, value)
        """
        if convertTime:
            return dict([(pandas.to_datetime(int(k), unit='s'), v) for k, v in self.dps.items()])

        return dict([(int(k), v) for k, v in self.dps.items()])


class OpenTSDBResponse(object):
    """ Complete OpenTSDB response """

    def __init__(self, otsdbResp):
        """
            Params:
                otsdbResp : raw opentsdb response as a str, list or tuple.
        """
        if isinstance(otsdbResp, str) or isinstance(otsdbResp, unicode):
            # string response
            self._series = [ OpenTSDBResponseSerie(**s) for s in json.loads(otsdbResp) ]
        elif isinstance(otsdbResp, list) or isinstance(otsdbResp, tuple):
            # dict response
            self._series = [ OpenTSDBResponseSerie(**s) for s in otsdbResp ]
        else:
            raise RuntimeError("Invalid type: %s" % (type(otsdbResp)))


    @property
    def series(self):
        """
            Use iterator for better memory management
        """
        for s in self._series:
            yield s


    def DataFrame(self, aliasTransform=None, convertTime=False):
        """
            Converts an OpenTSDB array response into a DataFrame
            Params:
                convertTime : Whether to convert epoch to pandas datetime
                aliasTransform : lambda function or string format to customize
                                 serie name i.e. alias
            Return:
                OpenTSDB response DataFrame
        """
        if aliasTransform == None:
            return pandas.DataFrame(dict([
                (s.id, s.datapoints(convertTime)) for s in self.series ]))
        else:
            return pandas.DataFrame(dict([
                (s.alias(aliasTransform), s.datapoints(convertTime)) for s in self.series ]))

class BaseClient(object):

    def __init__(self, host, port=4242, ssl=False):
        if ssl:
            self.url = "https://%s:%d" % (host, port)
        else:
            self.url = "http://%s:%d" % (host, port)

    def queryUrl(self, **kwargs):
        return str("%s/api/query?%s" % (self.url, self.__urlEncodedParams(**kwargs)))

    def __urlEncodedParams(self, aggr="sum", rate=False, counter=False, end=None, **kwargs):

        timeStr = "start=%s" % (kwargs["start"])
        if end != None:
            timeStr += "&end=%s" % (end)

        if rate:
            prefix = "%s:rate:%s" % (aggr, kwargs["metric"])
        elif counter:
            prefix = "%s:rate{counter,,1}:%s" % (aggr, kwargs["metric"])
        else:
            prefix = "%s:%s" % (aggr, kwargs["metric"])

        # TODO: check
        tagsStr = ",".join([ "%s=%s" % (k, kwargs["tags"][k]) for k in sorted(kwargs["tags"].keys()) ])

        if tagsStr != "":
            return "%s&m=%s{%s}" % (timeStr, prefix, tagsStr)
        else:
            return "%s&m=%s" % (timeStr, prefix)

class Client(BaseClient):

    def query(self, **kwargs):
        resp = requests.get(self.queryUrl(**kwargs))
        if resp.status_code >= 200 and resp.status_code < 400:
            return OpenTSDBResponse(resp.text)
            #return resp.text
        # error
        return resp.text