from fab.types import XPoints, LogProbFunc

class TransitionOperator(object):
    def transition(self, x: XPoints, log_q_x: LogProbFunc, i: int) -> XPoints:
        """Returns x generated from transition with log_q_x, as the invariant
        distribution."""
        return NotImplemented
