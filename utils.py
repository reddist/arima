from datetime import datetime


def add_months(date, months):
    new_month = date.month - 1 + months
    new_year = date.year + new_month // 12
    return datetime(new_year, (new_month % 12) + 1, 1)

