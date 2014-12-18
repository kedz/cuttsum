from datetime import datetime, timedelta
from collections import namedtuple
import os

def gen_dates(start, end):
    """Generate "YYYY-MM-DD-HH" format timestamps
    between start and end date or datetime objects
    at hourly intervals."""
    cur_date = start
    while cur_date <= end:
        yield cur_date.strftime('%Y-%m-%d-%H')
        cur_date += timedelta(hours=1)

DatetimeInterval = namedtuple('DatetimeInterval', ['start', 'stop'])
def hour_str2datetime_interval(hour, hours=1):
    start = datetime.strptime(hour, u'%Y-%m-%d-%H')
    stop = start + timedelta(hours=hours)
    return DatetimeInterval(start, stop) 
