'''
meter.py

Auxiliary meters.
'''

class AverageMeter(object):
    '''
    Computes and stores the average and current values.
    '''

    def __init__(self):
        '''
        Initializes AverageMeter
        '''
        self.reset()

    def reset(self):
        '''
        Resets the meter.
        '''
        self.var = 0 # current measured value
        self.avg = 0 # moving average value
        self.sum = 0 # running sum value
        self.cnt = 0 # sample counter

    def update(self, val, n=1):
        '''
        Updates the running statistics,

        @param val measured value.
        @param n number of measurements.
        '''

        # Computing
        self.val = val
        self.sum += val*n
        self.cnt += n
        self.avg = self.sum / self.cnt
