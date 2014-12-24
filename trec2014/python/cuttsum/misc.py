import sys

class ProgressBar:
    def __init__(self, max_jobs):
        self.max_jobs_ = max_jobs
        self.n_job_ = 0
        self.term_width_ = 70
        self.bin_size_ = max_jobs / float(self.term_width_)
    def update(self):
        self.n_job_ += 1
        if self.n_job_ == self.max_jobs_:
            sys.stdout.write(' ' * 79 + '\r')
            sys.stdout.flush()
        else:
            bins = int(self.n_job_ / self.bin_size_)
            bins = min(bins, self.term_width_)

            if bins > 0:
                bar = ' [' + '=' * (bins - 1) + '>' + \
                    ' ' * (self.term_width_ - bins) + ']\r'
            else:
                bar = ' [' + ' ' * self.term_width_ + ']\r'

            per = ' {:3.0f}% '.format(self.n_job_ * 100. / self.max_jobs_)
            bar = bar[:36] + per + bar[42:]

            sys.stdout.write(bar)
            sys.stdout.flush()

    def clear(self):
        sys.stdout.write('\r')
        sys.stdout.write(' ' * 79)
        sys.stdout.write('\r')
        sys.stdout.flush()
