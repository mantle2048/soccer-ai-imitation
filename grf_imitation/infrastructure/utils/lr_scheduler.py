from typing import Dict, List, Any
import torch
"yanked from https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/lr_scheduler.py"

"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
class Schedule:

    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()

    def __call__(self, t: int) -> Any:
        """Simply calls self.value(t). Implemented to make Schedules callable."""
        return self.value(t)

class ConstantSchedule(Schedule):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v

def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class PiecewiseSchedule(Schedule):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints
        if self._outside_value is None:
            self._outside_value = endpoints[-1][-1]

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(Schedule):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class MultipleLRSchedulers(dict):
    """A wrapper for multiple learning rate schedulers.
    Every time :meth:`~tianshou.utils.MultipleLRSchedulers.step` is called,
    it calls the step() method of each of the schedulers that it contains.
    Example usage:
    ::
        scheduler1 = ConstantLR(opt1, factor=0.1, total_iters=2)
        scheduler2 = ExponentialLR(opt2, gamma=0.9)
        scheduler = MultipleLRSchedulers(scheduler1, scheduler2)
        policy = PPOPolicy(..., lr_scheduler=scheduler)
    """

    def __init__(self, **kwargs: Dict[str, torch.optim.lr_scheduler.LambdaLR]):
        dict.__init__(self, **kwargs)
        self._itr = 0

    def __repr__(self):
        """Return str(self)."""
        lrs = self.get_last_lr()
        if not lrs:
            return "No scheduler"
        self_str = f"Itr: {self._itr}" + ":\n"
        for name, sche in self.items():
            lr = sche.get_last_lr()[0]
            self_str += f"{name}: {lr}, {sche.lr_lambdas[0]._endpoints} \n"
        return self_str

    def step(self) -> None:
        """Take a step in each of the learning rate schedulers."""
        for scheduler in self.values():
            scheduler.step()
        self._itr += 1

    def get_last_lr(self, with_suffix=True):
        lr_dict = {}
        for name, scheduler in self.items():
            if with_suffix:
                name = name + ' Learning Rate'
            lr_dict[name] = scheduler.get_last_lr()[0]
        return lr_dict

    def state_dict(self) -> List[Dict]:
        """Get state_dict for each of the learning rate schedulers.
        :return: A list of state_dict of learning rate schedulers.
        """
        return {n: s.state_dict() for n, s in self.items()}

    def load_state_dict(self, state_dict: Dict[str, Dict]) -> None:
        """Load states from state_dict.
        :param List[Dict] state_dict: A list of learning rate scheduler
            state_dict, in the same order as the schedulers.
        """
        assert state_dict.keys() == self.keys()
        for (s, sd) in zip(self.values(), state_dict.values()):
            s.__dict__.update(sd)
