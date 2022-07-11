from time import time

def timer(func):
    def deco(*args, **kwargs):
        tik = time()
        res = func(*args, **kwargs)
        tok = time()

        print(f'Timer: function `{func.__name__}` running for '
              f'{round(tok-tik, 4)}s ({round((tok-tik)/60, 4)} min).')
        return res

    return deco