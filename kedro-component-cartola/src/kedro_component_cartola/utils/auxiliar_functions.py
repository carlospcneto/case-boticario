from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def back_in_time(date: str, days: int = 0, months: int = 0, years: int = 0) -> str:
    """
    Returns a date string that is a specified number of days, months, and years before the given date.
    
    Args:
        date (str): The date in 'YYYY-MM-DD' format.
        days (int): Number of days to subtract.
        months (int): Number of months to subtract.
        years (int): Number of years to subtract.
    
    Returns:
        str: The new date in 'YYYY-MM-DD' format.
    """
    dt = datetime.strptime(date, '%Y-%m-%d')
    dt -= timedelta(days=days)
    dt -= relativedelta(months=months, years=years)
    
    return dt.strftime('%Y-%m-%d')