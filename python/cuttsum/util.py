from datetime import timedelta


def gen_dates(start, end):
    """Generate "YYYY-MM-DD-HH" format timestamps
    between start and end date or datetime objects
    at hourly intervals."""
    cur_date = start
    while cur_date <= end:
        yield cur_date.strftime('%Y-%m-%d-%H')
        cur_date += timedelta(hours=1)

