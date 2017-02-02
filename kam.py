import numpy as np
from pykalman import KalmanFilter


class KalmanAggModel(object):
    """
    Kalman filter to aggregate predictive models in time series.
    
    Given a single value to predict and multiple (somewhat independent)
    models to predict it, this tool uses a kalman filter to aggregate
    predictions more accurately and with less variance than a simple 
    mean of predictions, with addded benfit of estimating the 
    variance of the aggregate prediction.
    
    Note:
        When variance of prediction errors are less than variance
        of the target this tool provides little benefit over simple 
        mean of predictions.
    
    Attributes:
        kf (KalmanFilter): The pykalman standard Kalman filter model.
    """
    def __init__(self, obs, target, init_state_mean, init_state_cov=1.):
        """
        Initialize model with appropriate settings.
        
        Args:
            obs (ndarray): An array of predicted values.
                Array width indicates number models to be aggregated. 
            target (ndarray): An array of the target variable values. 
            init_state_mean (float): Initial estimate of target mean.
            init_state_cov (float):  Initial estimate of target variance.
                `init_state_cov` defaults to 1.
                
        """
        for i in (obs, target):
            if not isinstance(i, np.ndarray):
                raise TypeError('obs and target must be numpy ndarrays')
        
        if self.ensure_2d(target).shape[1] > 1:
            raise Exception('only single variable targets accepted')
        
        init_sm = np.array([init_state_mean,])
        init_scov = np.eye(1) * init_state_cov
        
        # transition assumes random walk
        trans_mat = np.eye(init_sm.shape[0])
        trans_cov = np.eye(init_sm.shape[0]) * np.var(target)
        
        # observations assumed forecasts of target
        sample_ob = self.ensure_2d(obs)[0]
        obs_mat = np.ones([sample_ob.shape[0], 1]) 
        obs_cov = self.ensure_2d(self.calc_obs_cov(obs, target)) 
        
        # checks that matrix shapes are aligned
        groups = ((init_scov, init_sm), (init_scov, init_sm), (obs_cov, sample_ob))
        for cov, mean in groups:
            for ind in (0,1):
                assert cov.shape[ind] == mean.shape[0]
        
        groups = ((init_sm, trans_mat, init_sm), (sample_ob, obs_mat, init_sm))
        for y, A, x in groups:
            assert np.dot(A, x).shape == y.shape
        
        self.kf = KalmanFilter(n_dim_obs=sample_ob.shape[0], 
                               n_dim_state=init_sm.shape[0],
                               initial_state_mean=init_sm,
                               initial_state_covariance=init_scov,
                               transition_matrices=trans_mat,
                               observation_matrices=obs_mat,
                               observation_covariance=obs_cov,
                               transition_covariance=trans_cov,)
        
    def ensure_2d(self, x):
        if not len(x.shape):
            return np.array([[x]])
        if len(x.shape) == 1:
            return  x[:, np.newaxis]
        return x
    
    def calc_obs_cov(self, obs, target):
        return np.cov((obs.T - target.T))
        
    def fit(self, obs):        
        """Estimates state mean and cov after each given observation.

        Args:
            obs (ndarray): An array of predicted values.
                Array width indicates number models to be aggregated. 
    
        Returns:
            2d array corresponding to estimated mean and variance

        """  
        if not isinstance(obs, np.ndarray):
            raise TypeError('obs must be numpy ndarrays')
        
        if self.ensure_2d(obs).shape[1] != self.kf.n_dim_obs:
            raise Exception('obs must have same num columns as when initialized')
        
        res = self.kf.filter(obs)
        res = map(lambda x: x.flatten(), res)
        return np.array(res).T
    
    def online_fit(self):
        # self.kf.filter_update()
        raise Exception('not yet implemented')