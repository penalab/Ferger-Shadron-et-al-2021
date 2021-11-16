"""
"""

import numpy as np
from base64 import decodebytes as b64_decodebytes

class ModelRunnerBase():
    
    def __init__(self, *, bcs = None, sds = None, n_trials = 500):
        
        # Number of trials to run
        self.n_trials = n_trials
        
        # Stimulus directions (deg):
        self.azimuths = np.array([-75, -55, 55, 75])
        
        # For conversion between azimuth angles and ITD:
        # ITD (us) at 90 deg angle
        self.A = 260.0
        # 
        self.w = 0.0143
        
        # BC, binaural correlation - IMPORTANT: ranges from 0.0 to 1.0
        if bcs is None:
            # 2011 model BCs:
            self.bcs = np.array([10, 20, 30, 40, 50, 70, 100], dtype=float)
        else:
            # arbitrary BCs:
            self.bcs = np.asarray(bcs, dtype=float)
        
        if sds is None:
            # 2011 model standard deviations:
            self.sds = 219.34 * np.exp(-0.1131 * self.bcs) + 41.2
        else:
            # arbitrary SDs:
            self.sds = np.asarray(sds, dtype=float)
            if not self.sds.size == self.bcs.size:
                raise ValueError("An array-like of arbitrarily set SDs must "
                                 "have the same size as the BCs.")
    
    def calc_vec_lengths(self, vary, Y):
        raise NotImplementedError("Model specific implementation needed. "
                                  "Use 'PVmodelRunner' or 'BayesModelRunner'")
    
    def run(self):
        # norm 1 vectors in cell preferred directions
        self.vectors = (
            np.cos(self.directions * np.pi / 180) +
            np.sin(self.directions * np.pi / 180) * 1j
        )
        
        # Results matrix:
        results = np.zeros((self.n_trials,self.azimuths.size,self.bcs.size))
        
        for j in range(self.bcs.size):
            # ITD noise variance
            vary = self.sds[j] ** 2
            for n in range(self.n_trials):
                for m in range(self.azimuths.size):
                    # Observed ITD
                    Y = self.A * np.sin(self.w * self.azimuths[m]) + np.random.randn() * self.sds[j]
                    # Model specific:
                    vec_lengths = self.calc_vec_lengths(vary, Y)
                    # population vector <=> posterior mean:
                    m_vector = np.sum( vec_lengths * self.vectors)
                    # direction of population vector
                    results[n, m, j] = np.angle(m_vector)
        # Get mean and standard deviation over trials
        self.m_rad = np.mean(results, axis=0)
        self.s_rad = np.std(results, axis=0)
        # Convert to degrees
        self.m = self.m_rad * 180 / np.pi
        self.s = self.s_rad * 180 / np.pi
        # Remember full results matrix:
        self.results = results


class PVmodelRunner(ModelRunnerBase):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ### PV model specific parameters
        # Max response
        self.Rmax = 10
        # cell preferred directions
        self.directions = PV_DIRECTIONS
        self.run()
    
    def calc_vec_lengths(self, vary, Y):
        # Population activities of the neurons on this trial
        R = self.Rmax * np.exp(-.5 / vary * (Y - self.A * np.sin(self.w * self.directions)) ** 2)
        # Poisson variability
        r = np.random.poisson(R)
        # normalize responses (if sum == 0, no normalization needed)
        if np.sum(r) > 0:
            r = r / np.sum(r)
        return r

class BayesModelRunner(ModelRunnerBase):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ### Bayes model specific parameters
        # prior variance (deg^2)
        self.varx = 542.0
        # direction variable for Bayesian model (deg)
        self.directions = np.linspace(-180, 180, 180*2*2+1)
        # prior (un-normalized)
        self.prior = np.exp(-.5 / self.varx * self.directions ** 2)
        # mean ITD (us) at each direction
        self.f = self.A * np.sin(self.w * self.directions)
        self.run()
    
    def calc_vec_lengths(self, vary, Y):
        # posterior = likelihood x prior
        p = np.exp(-.5 / vary * (Y - self.f) ** 2) * self.prior
        # normalize posterior
        p = p / np.sum(p)
        return p



# Cell preferred directions for PV model, n = 500 cells
# Data included here to avoid shipping separate file
PV_DIRECTIONS = np.frombuffer(b64_decodebytes(
b'3lxKUnG4UcATrlMVIx1PwKLnJIjWxk7AYc32EKc2TsBhDWZLtRxNwLfuxFp6vkzA5pRKM84kTMA/'
b'wWINS+tKwEvPWmnNsknAH8lN1cI7SMAyYB092vBGwNg9NNuBe0bASdRc93dfRsCuGQIH3jNGwBq7'
b'sIf0qEXA0tDfkpIWRcCjIAxqmPFEwJQeuX/o3kTApyljmHqYRMCkyJ+9JWNEwOM2oY/ihEPA7k/z'
b'ZQ5gQ8BG1N4M4C5DwAtOMFiWHkPAKc8DzhLmQsARl6Q48tpCwFrJc8/YYULABvBMPgA9QsDKarN5'
b'URdCwE5zh0vbFULAGhXA+BDUQcBfo3spHcdBwIRBWwX+pkHAd+Fe6mrHQMDwMtHSSsBAwPc51e9j'
b'vUDAPkTNrjaZQMDcXTiIJphAwBMnpJx5LEDAHZByLi0sQMDgMhxsrx1AwDYkim6a9j/AEKVgXxD1'
b'P8CVFWCm/9o/wAYk76MIyz/AbbskWicLP8BCcUVbwPo+wLxM7+qx8D7AQWns3shRPsBoifh0lE4+'
b'wIfOyhvXLj7ApIBvAjsWPsBRbdR5BAk+wPGreGSI+j3AGrEOz1YUPcDR6806vvI8wPjva70v5jzA'
b'KU4ZEYPpO8DSYoHuWiw7wCdUGlRs8jrATUvHmfrXOsBD/KFhL5o6wIj0WhNCZjrAU7kiR81XOsCj'
b'DfrzO/s5wF4bsqX35DnAiH6UNQjGOcAiwssXbLE5wH/t0Y0NnjnATZG1f44hOcDTcfOFtv84wAEH'
b'mtPohjjAK+C3Bz53OMA8XrgefHI4wAs/0puwSTjAufPrD508OMB/GTICLgs4wGbzKubg8DfAyeY/'
b'ijTSN8Bs+AeJrdE3wF1TTpFIyjfAK8UgOjGTN8AMNHcaLog3wDFtIndXhDfAjYigVL9sN8AXuz2y'
b'BUE3wBVFkwhkOTfAZcgsu6YaN8DPnjUuXNY2wPuB/PNH1TbA/2XuKq5ZNsBCJ8jlX1Q2wPZKyj2B'
b'RDbApKzNY3NDNsDcY2M+6ik2wPwO18YryDXA2EGxH7SMNcD/Myz6E301wDfkka57aDXAigM5syxO'
b'NcAFlKCSCnA0wFFNbInKbTTAIAYNULODM8AEWAc5jmczwJJ1mZJpyzLAYIB2bDSuMsBYr0vPdKwy'
b'wOLpTVHlkDLA5lZkFDVtMsCQ2Tk3XF8ywKCc6yKWMjLAi5qlEyTxMcCVF3kBEOwxwAxM5LOH0DHA'
b'nD/79DLIMcDhjSNBULkxwL6b5qaVsjHAVzb69F2vMcDjnHRhhIcxwCAvS4pSFzHASfAEJsbWMMAI'
b'kOMCstYwwJfa9IJXkTDAollkUfd3MMCgMXGXfEwwwPdviU6AEjDAVWjMGk7sL8CCTlrxDOkvwGZL'
b'dg1D5S/AuqYOtnqQL8CoQDYJNBkvwG4jHLksES/AOpyFfSRZLsAiq8aL4dYtwHDPC139eS3AfZ0+'
b'iK1qLcBm91USPystwGVZ7gsGXSzAsDIYEHw8LMCTpgKLuFgrwG19Gyl3tyrAIx17hJ9AKsDk0Dra'
b'pTQqwJyFGTe0GyrAoS7lL1sZKsCHeTKiFZQpwKmMNg4yJinAdY8oOtocKcDGZTZ8MIIowPNJ3oRw'
b'uCfAJMRrEC6sJ8BiEYJkKFwnwC4CrKRrTyfAWn7LNaMwJ8DwJB9LhiQnwBN1Ol2dPibAgenGOo06'
b'JsAvPDCE6ycmwIfHJbVuICbAXECC0QDwJcCmjTrc88olwJBEjPOenSXAgQcANdiaJcCnKhRMCpol'
b'wE7gAIDniCXATObI7/dNJcD/pNlUTkglwLp85bFOKyXAWucFw4ckJcCcGXfpzPwkwNM+iClAiCTA'
b'fno2fpN0JMDLn3SlRLcjwKZR8ff2gyPAJicFej55I8DP0f+ZwEcjwAA3E7aCGSPAEC6iZyQBI8BE'
b'9dT0RM0iwKNZ9Z5UciLANLA27SPvIcDYDkYh538hwIYMibJ7TSHAscumLtM1IcD1Gh4db40gwCVp'
b'rJmPAiDAXKfTK7z4H8DokCAce+kfwJ+iCH1gWB/AHu4PkIvCHsA2eZsjaHIewMCoadkbwx3ATc87'
b'ToQZHcBBRK8V4GIbwAGA+SzGWRrArYtFblWCGcDLtEQldjAZwGPonI4XEhnAeN9LaYTcGMA8FjhD'
b'p6AYwOfcV56tTxjAeOJyg4F+F8Bj4k/55hsXwHL265goERfAbD7t/yCGFsAmjIecAlQWwNUjEziH'
b'oRXAas3PJZVGFcAB4dI4jwsVwCNvv1MjlhTA+nEsvAfxEsCVg7pCRp0SwJLo7XRFohHAwQayE6oW'
b'EcCv8oSSV+0QwKLQz0Jd9Q/Asnga+7RlD8B+32M2gEsPwLASYF7qTg7AyOz1IgZzDcBdfUJIZ40L'
b'wEmr6f93GAvAY0xSiUA7CMAIwNANha4HwBc6am6alAfAqVI5tpoTB8C1S7qVce0GwCBJ5drWsgPA'
b'QxoXZm1KAsDuW8clIxICwC4h102ByQDAKb5BHi0o/r9XU1yktwf6v1pyXmrbzfm/5znVk1es+L8A'
b'o1PEnVnzv4HVu95NU/K/r4XyKxIK8r+/DOQRWsrxvxmY+TNqC/G/KBLWjsep6r9MVSCyabvcvzOu'
b'IOBK69e/fpqAMxLA0r/9OFSw2NDQv1TQL5bi+M6/HraxV1dDmr8tqvvQPiNGP2QTpT+518E/v7tf'
b'dK7UyD8TLhacYVTMPygkZ3VSi8w/nzjvzcI/0T+xfaHEaFnTP4POrP28t+A/akzhbfXD4D+Kvxw1'
b'N3vjP01wvq3bNuw/ZK8iI8HN7D+EZkCB+PztPyOz0CycmfA/96E1E37n8D8eqSH5S+jwPwGRAQwt'
b'FvE/kRPJuse/9D+/jr4UYNL0P7Jopu7suvc/Fxs+4doE+D8zBbZFWJD5P2QvR8aaff4/1rvsPHvK'
b'/j9KaPyVhPD/PyuEJOyT+P8/HSA1G88NAEBiMeWD9xQAQMED44WcPQBAE8X33qpXAEAsN7C9OFoB'
b'QHdaGK9w8QFAU4ZRVpCfBEAttpKkJ0YFQKHKMTgJJQZAMvzRPhyMBkByd/RAfFMHQOIvCNMhpgdA'
b'IUpxJsVPCEDo29QbehILQP2jMYejPAtAuI32WwQODEAynOJlQDEMQDj4FPZzrhBAj8MlaQrXEEBM'
b'sAiKzgQRQMI6PTLCwBNAn4UjwmM4FECplubYS8YUQHPUnrA26xRAlfqROXE/FkBumc86vFQWQCKj'
b'XWYH6xZACtmPy2NtF0BvybMlP8kXQPtVSfhd3RdA6T9HmCc0GECwj94GoaEZQBLR27ITyBlAXIJp'
b'vOi4GkBSFaSOphobQKVGBiMfWRtAIL5vz90NHEDk25HEtc8cQAlf1Jhw4RxAZq9fOLqjHUC9SoOX'
b'//kdQJS/oH1EAB5ALhaMJvxQHkC8FEiGuWkeQPsB+dQQgR5A6WDuYnC4HkBuJOusfN0fQAA0xON6'
b'/R9AL/VWbhAAIEBjNa5+DDQgQKO7grIJQSBA6lk9qfvwIEA9xxfU4E8hQNNeJCx1jyFAturHhhWR'
b'IUAJ2SeF7pMhQCtOFn8/6iFA27NEjRj9IUA76InNUP8hQIrQ8VuhoyJALDa7UwIFI0CWNYYUKSgj'
b'QBSMC0fNKCNA/ds4AcwwI0BcnZuYJEgjQJ/19M+NgCNA9+WX7m6/I0APfKj6fREkQJMPhvWmLyRA'
b'oNjP7uY3JEDFvM8zVEwkQHA/x6WumCRAhNJIZftDJUAcOc/0f6clQI1cy9gZuiVARIk1lMK6JUDN'
b'XWjLq/MlQN+SG7oxfiZAbGywRFStJkBRpfxA2UInQP4nOTtiDihApb/LYolVKEC5OjgHZo0oQH1a'
b'EihMyyhAEVF6iJMMKUAqkPexoDkpQFXpXRoNYClAu9prwOZwKUADqIRQoPQpQP7mTC+J+ClAUKGP'
b'7Q1aKkB751FCVm8qQEJxLWWShCpAJHFEJIlBK0D8+DGnklwrQCt4kV4LxStAPCFDO3PHK0CQW0qu'
b'pBgsQIkDClwgZyxAPAZ/xTjCLEBtnQXmrNAsQESTG9CqWy1AOETQWlCQLUDrm9lmTFQuQIxCzxw6'
b'ly5AeEd1oG2bLkB/51ZNoaEuQOzSFFmCwC5ARRF7uuHeLkBl+v2QggEvQNuCqwC/xC9AlCiZVIQw'
b'MED1v2+tjkwwQLdPEIYDdTBAwc5vH1KFMEDf9jVzW4YwQHaJxSSWnDBA0oK5eX3iMEBjKieowuUw'
b'QNK6EBBu9TBAjRfRFVaNMUAADyunxrIxQFK4xKS6tTFAWADsklTWMUCceSRaGeUxQLW1DiNlSTJA'
b'8He4KqRqMkCHRMPMZHQyQG1iVk8KjDJAqBGLysShMkBmRumpFYwzQHszzO5pkDNAbzW+5hPNM0C9'
b'5MLhHfAzQDFhBkvklzRAx/m42fCwNEBxMIsenro0QFuUPswbHTVA/yNkOEE8NUAs4kl/en01QDs9'
b'lBXUhzVANEVjFnqpNUACthY1jbQ1QKYRVVXFOzZArxeNEuanNkCVZqEMGbU2QAR8J0GhezdAM10u'
b'itOnN0BJvYITkbE3QKoh3WkNtzdAdj9PDj68N0BlUNCi78g3QBXNui7TKThA8DO6qttDOECIEzm0'
b'S2E4QJsTZamNdzhAuxE2QPt4OEBuB+SHnZY4QEiCp00c0zhAFic87/zcOEAfrEYHFlE5QCLRaEpS'
b'XjlApUmy6tmqOUAak+5zsdo5QDdq/TeK5TlAZHss9NknOkDRvmtBW086QPuklMRfUTpAJmJzmw6O'
b'OkCJK1Dxkr46QKvImVhYTjtApkTbSqJ8O0BN/YYxUZ87QNP7+WY8tjtAycWs8c8NPEDu4ThKvEY8'
b'QASVLJtzfzxAWJzbeKPaPEDVIgnRZOE8QGMvnpDv5jxAEgTmwz/oPECRnAbxHfM8QBXNroDg/TxA'
b'4RURRAlkPUDrgw+MQmk9QMh9YO8Y4D1ArhQnRfpRPkCQByVrSY0+QLUjHf/ipj5A7E5OOMmXP0A/'
b'oT4dpIZAQM1tLznorkBA4aPT7inKQEBuHHTyNDZBQBdZ3ZLoUEFA3Ua2lJKuQUCmP+D1LvVBQI0g'
b'vi8SJkJASsjSGC9cQkC352JR4YtCQPLzyC2gjkJA8D9oD8SlQkDhJyg5t/1CQAE3/GUOT0NA7klD'
b'6MRVQ0ChF1umZn1DQI2Ho+sznUNACjgAQYu7Q0By45jicBBEQAXLmmcSYURA6RMzn2BrREC2KRab'
b't6JEQB/6i/cDPEVAs8VjwTBTRUCh7Vu4/uxFQDQ8CQPKaEZA7+F4vd1xRkAw0MRCoaRGQBIqIHff'
b'zEZA0arMtoTfRkBHi7pGOX9IQHymL4eNskhAzQ+52OhAS0Cs2SPWhf9OQIRmKnxWElBAp9ukBjNL'
b'UUBaECfC809RQA=='))
