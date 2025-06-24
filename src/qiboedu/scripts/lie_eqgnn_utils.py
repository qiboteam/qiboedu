from math import sqrt
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from sklearn.metrics import roc_curve

"""Some auxiliary functions"""

# Speed of light (m/s)
c = 299792458

"""Lorentz transformations describe the transition between two inertial reference
frames F and F', each of which is moving in some direction with respect to the
other. This code only calculates Lorentz transformations for movement in the x
direction with no spatial rotation (i.e., a Lorentz boost in the x direction).
The Lorentz transformations are calculated here as linear transformations of
four-vectors [ct, x, y, z] described by Minkowski space. Note that t (time) is
multiplied by c (the speed of light) in the first entry of each four-vector.

Thus, if X = [ct; x; y; z] and X' = [ct'; x'; y'; z'] are the four-vectors for
two inertial reference frames and X' moves in the x direction with velocity v
with respect to X, then the Lorentz transformation from X to X' is X' = BX,
where

    | γ  -γβ  0  0|
B = |-γβ  γ   0  0|
    | 0   0   1  0|
    | 0   0   0  1|

is the matrix describing the Lorentz boost between X and X',
γ = 1 / √(1 - v²/c²) is the Lorentz factor, and β = v/c is the velocity as
a fraction of c.
"""


def beta(velocity: float) -> float:
    """
    Calculates β = v/c, the given velocity as a fraction of c
    >>> beta(c)
    1.0
    >>> beta(199792458)
    0.666435904801848
    """
    if velocity > c:
        raise ValueError("Speed must not exceed light speed 299,792,458 [m/s]!")
    elif velocity < 1:
        # Usually the speed should be much higher than 1 (c order of magnitude)
        raise ValueError("Speed must be greater than or equal to 1!")

    return velocity / c


def gamma(velocity: float) -> float:
    """
    Calculate the Lorentz factor γ = 1 / √(1 - v²/c²) for a given velocity
    >>> gamma(4)
    1.0000000000000002
    >>> gamma(1e5)
    1.0000000556325075
    >>> gamma(3e7)
    1.005044845777813
    >>> gamma(2.8e8)
    2.7985595722318277
    """
    return 1 / sqrt(1 - beta(velocity) ** 2)


def transformation_matrix(velocity: float) -> np.ndarray:
    """
    Calculate the Lorentz transformation matrix for movement in the x direction:

    | γ  -γβ  0  0|
    |-γβ  γ   0  0|
    | 0   0   1  0|
    | 0   0   0  1|

    where γ is the Lorentz factor and β is the velocity as a fraction of c
    >>> transformation_matrix(29979245)
    array([[ 1.00503781, -0.10050378,  0.        ,  0.        ],
           [-0.10050378,  1.00503781,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    return np.array(
        [
            [gamma(velocity), -gamma(velocity) * beta(velocity), 0, 0],
            [-gamma(velocity) * beta(velocity), gamma(velocity), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )



def unsorted_segment_sum(data, segment_ids, num_segments):
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Adapted from https://github.com/vgsatorras/egnn.
    '''
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    Adapted from https://github.com/vgsatorras/egnn.
    '''
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def normsq4(p):
    r''' Minkowski square norm
         `\|p\|^2 = p[0]^2-p[1]^2-p[2]^2-p[3]^2`
    ''' 
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)
    
def dotsq4(p,q):
    r''' Minkowski inner product
         `<p,q> = p[0]q[0]-p[1]q[1]-p[2]q[2]-p[3]q[3]`
    '''
    psq = p*q
    return 2 * psq[..., 0] - psq.sum(dim=-1)

def normA_fn(A):
    return lambda p: torch.einsum('...i, ij, ...j->...', p, A, p)

def dotA_fn(A):
    return lambda p, q: torch.einsum('...i, ij, ...j->...', p, A, q)
    
def psi(p):
    ''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)`
    '''
    return torch.sign(p) * torch.log(torch.abs(p) + 1)





##############################


def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def args_init(args):
    r''' Initialize seed and exp_name.
    '''
    if args.seed is None: # use random seed if not specified
        args.seed = np.random.randint(100)
    if args.exp_name == '': # use random strings if not specified
        args.exp_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    if (args.local_rank == 0): # master
        print(args)
        makedir(f"{args.logdir}/{args.exp_name}")
        with open(f"{args.logdir}/{args.exp_name}/args.json", 'w') as f:
            json.dump(args.__dict__, f, indent=4)

def sum_reduce(num, device):
    r''' Sum the tensor across the devices.
    '''
    if not torch.is_tensor(num):
        rt = torch.tensor(num).to(device)
    else:
        rt = num.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        warmup_epoch: target learning rate is reached at warmup_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    Reference:
        https://github.com/ildoonet/pytorch-gradual-warmup-lr
    """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    @property
    def _warmup_lr(self):
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch + 1) / self.warmup_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * (self.last_epoch + 1) / self.warmup_epoch + 1.) for base_lr in self.base_lrs]

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch - 1:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return self._warmup_lr

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        self.last_epoch = self.last_epoch + 1 if epoch==None else epoch
        if self.last_epoch >= self.warmup_epoch - 1:
            if not self.finished:
                warmup_lr = [base_lr * self.multiplier for base_lr in self.base_lrs]
                for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                    param_group['lr'] = lr
                self.finished = True
                return
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.warmup_epoch)
            return

        for param_group, lr in zip(self.optimizer.param_groups, self._warmup_lr):
            param_group['lr'] = lr

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.warmup_epoch)
                self.last_epoch = self.after_scheduler.last_epoch + self.warmup_epoch + 1
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        result = {key: value for key, value in self.__dict__.items() if key != 'optimizer' or key != "after_scheduler"}
        if self.after_scheduler:
            result.update({"after_scheduler": self.after_scheduler.state_dict()})
        return result

    def load_state_dict(self, state_dict):
        after_scheduler_state = state_dict.pop("after_scheduler", None)
        self.__dict__.update(state_dict)
        if after_scheduler_state:
            self.after_scheduler.load_state_dict(after_scheduler_state)