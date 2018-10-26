'''
Helper code for visualising progress in the terminal.
'''

def bar(cur_value, max_value, prefix='', suffix='', length=40):
    '''Print a progress bar to stdout.

    Args:
        cur_value (int): current iteration number.
        max_value (int): total iterations.
        prefix (str): prefix string.
        suffix (str): suffix string.
        length (int): progress bar length.
    '''

    percent = ('{:6.2f}').format((100.0 * cur_value) / max_value)
    bar_length = int((length * cur_value) // max_value)
    bar_str = '#' * bar_length + '.' * (length - bar_length)
    out_str = '\r{} [{}] {}% {}'.format(prefix, bar_str, percent, suffix)
    if cur_value < max_value:
        print(out_str, end='\r')
    else:
        print('\r\033[K', end='\r')
