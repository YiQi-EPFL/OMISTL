from abc import ABC, abstractmethod

class Problem(ABC):
    """Abstract class to handle all problem instances"""
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def init_problem(self): 
        """Abstract method to implement problem in cvxpy."""
        raise NotImplementedError

        
class Solver(ABC):
    """Abstract class to handle all solver instances."""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, params):
        """Abstract method mapping parameter values to predicted strategy.
        
        Args:
            params: (list of) numpy array(s) of parameter values for specific
                problem instance.
            
        Returns:
            y_hat: numpy array of ints, estimate of optimal strategy, of shape
                [self.n_evals, self.n_y].  
        """
        raise NotImplementedError