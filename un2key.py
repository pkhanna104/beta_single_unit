
def convert(unit):
    if len(unit) == 2:
        key = 'sig00'+unit
    elif len(unit) == 3:
        key = 'sig0'+unit
    elif len(unit) == 4:
        key = 'sig'+unit
    else:
        raise

    return key