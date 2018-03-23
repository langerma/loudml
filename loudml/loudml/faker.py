import loudml.vendor

import argparse
import datetime
import logging
import random
import time

from . import (
    errors,
)
from .misc import (
    make_datetime,
)
from .randevents import (
    CamelEventGenerator,
    LoudMLEventGenerator,
    FlatEventGenerator,
    SinEventGenerator,
)

def generate_data(ts_generator, from_date, to_date):
    for ts in ts_generator.generate_ts(from_date, to_date, step=60):
        yield ts, {
            'foo': random.lognormvariate(10, 1),
        }

def dump_to_json(generator):
    import json

    data = []

    for ts, entry in generator():
        entry['timestamp'] = ts
        data.append(entry)

    print(json.dumps(data,indent=4))

def build_tag_dict(tags=None):
    tag_dict = {}
    if tags:
        for tag in tags.split(','):
            k, v = tag.split(':')
            tag_dict[k] = v
    return tag_dict

def dump_to_influx(generator, addr, db, measurement, tags=None, clear=False):
    from .influx import InfluxDataSource

    source = InfluxDataSource({
        'name': 'influx',
        'addr': addr,
        'database': db,
    })

    if clear:
        source.drop()
    source.init()

    for ts, data in generator:
        now = time.time()
        if ts > now:
            time.sleep(ts - now)

        source.insert_times_data(
            measurement=measurement,
            ts=ts,
            data=data,
            tags=tags,
        )

def dump_to_elastic(generator, addr, index, doc_type, tags=None, clear=False):
    from .elastic import ElasticsearchDataSource

    source = ElasticsearchDataSource({
        'name': 'elastic',
        'addr': addr,
        'index': index,
    })

    if clear:
        source.delete_index()

    properties = {
        "timestamp": {
            "type": "date"
        },
        "foo": {
            "type": "float",
        },
    }

    if tags:
        for k in tags.keys():
            properties[k] = {
                "type": "keyword",
            }

    source.create_index({
        "template": index,
        "mappings": {
            doc_type: {
                "include_in_all": true,
                "properties": properties,
            },
        }
    })

    for ts, data in generator:
        now = time.time()
        if ts > now:
            time.sleep(ts - now)

        data.update(tags)
        source.insert_times_data(
            measurement=measurement,
            ts=ts,
            data=data,
            doc_type=doc_type,
        )

def main():
    """
    Generate dummy data
    """

    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        choices=['json', 'influx', 'elastic'],
        default='json',
    )
    parser.add_argument(
        '-a', '--addr',
        help="Output address",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        '-i', '--index',
        help="Index",
        type=str,
    )
    parser.add_argument(
        '-b', '--database',
        help="Database",
        type=str,
        default='dummy_db',
    )
    parser.add_argument(
        '-m', '--measurement',
        help="Measurement",
        type=str,
        default='dummy_data',
    )
    parser.add_argument(
        '--from',
        help="From date",
        type=str,
        default="now-7d",
        dest='from_date',
    )
    parser.add_argument(
        '--to',
        help="To date",
        type=str,
        default="now",
        dest='to_date',
    )
    parser.add_argument(
        '--shape',
        help="Data shape",
        choices=['flat', 'sin', 'camel', 'loudml'],
        default='sin',
    )
    parser.add_argument(
        '--avg',
        help="Average rate",
        type=float,
        default=5,
    )
    parser.add_argument(
        '-c', '--clear',
        help="Clear database or index before insertion "\
             "(risk of data loss! Use with caution!)",
        action='store_true',
    )
    parser.add_argument(
        '--tags',
        help="Tags",
        type=str,
    )

    arg = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if arg.shape == 'flat':
        ts_generator = FlatEventGenerator(avg=arg.avg)
    elif arg.shape == 'loudml':
        ts_generator = LoudMLEventGenerator()
    elif arg.shape == 'camel':
        ts_generator = CamelEventGenerator()
    else:
        ts_generator = SinEventGenerator(avg=10, sigma=2)

    from_date = make_datetime(arg.from_date)
    to_date = make_datetime(arg.to_date)

    logging.info("generating data from %s to %s", from_date, to_date)

    generator = generate_data(ts_generator, from_date.timestamp(), to_date.timestamp())

    tag_dict = build_tag_dict(arg.tags)

    try:
        if arg.output == 'json':
            dump_to_json(data)
        elif arg.output == 'influx':
            dump_to_influx(
                generator,
                addr=arg.addr,
                db=arg.database,
                clear=arg.clear,
                measurement=arg.measurement,
                tags=tag_dict,
            )
        elif arg.output == 'elastic':
            pass
    except errors.LoudMLException as exn:
        logging.error(exn)