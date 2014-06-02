from datetime import timedelta


def gen_dates(start, end):
    """Generate "YYYY-MM-DD-HH" format timestamps
    between start and end date or datetime objects
    at hourly intervals."""
    cur_date = start
    while cur_date <= end:
        for h in range(24):
            yield '{}-{:02}'.format(cur_date, h)

        cur_date += timedelta(days=1)

