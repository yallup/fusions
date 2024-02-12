import logging

from fusions.cfm import CFM
from fusions.diffusion import Diffusion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from anesthetic import MCMCSamples, NestedSamples

# from anesthetic.read.hdf import read_hdf, write_hdf
from scipy.special import logsumexp
from tqdm import tqdm

from fusions.model import Model


@dataclass
class Point:
    x: np.ndarray
    logl: float
    logl_birth: float
    logl_pi: float = field(default=0.0)


@dataclass
class Stats:
    nlive: int = field(default=0)
    nlike: int = field(default=0)
    ndead: int = field(default=0)
    logz: float = field(default=-1e30)
    logz_err: float = field(default=1)
    logX: float = field(default=0)

    def __repr__(self):
        return (
            f"Stats(\n"
            f"  nlive: {self.nlive},\n"
            f"  nlike: {self.nlike},\n"
            f"  ndead: {self.ndead},\n"
            f"  logz: {self.logz},\n"
            f"  logz_err: {self.logz_err},\n"
            f"  logX: {self.logX}\n"
            f")"
        )


class Integrator(ABC):
    def __init__(self, prior, likelihood, **kwargs) -> None:
        self.prior = prior
        self.likelihood = likelihood
        self.logzero = kwargs.get("logzero", -np.inf)
        self.dead = []
        self.dists = []
        self.stats = Stats()
        self.model = kwargs.get("model", CFM)

    def sample(self, n, dist, logl_birth=0.0, beta=1.0):
        x = np.asarray(dist.rvs(n))
        logl = self.likelihood.logpdf(x) * beta
        self.stats.nlike += n
        logl_birth = np.ones_like(logl) * logl_birth
        log_pi = self.prior.logpdf(x)
        points = [
            Point(x[i], logl[i], logl_birth[i], logl_pi=log_pi[i])
            for i in range(x.shape[0])
        ]
        return points

    def stash(self, points, n, drop=False):
        live = sorted(points, key=lambda lp: lp.logl, reverse=True)
        if not drop:
            self.dead += live[n:]
        contour = live[n].logl
        live = live[:n]
        return live, contour

    @abstractmethod
    def run(self, **kwargs):
        pass

    @abstractmethod
    def update_stats(self, live, n):
        pass

    @abstractmethod
    def points_to_samples(self, points):
        pass

    def samples(self):
        return self.points_to_samples(self.dead)

    def importance_integrate(self, dist, n=1000):
        points = dist.rvs(n)
        likelihood = self.likelihood.logpdf(points)
        prior = self.prior.logpdf(points)
        return logsumexp(likelihood + prior) - np.log(n)

    def write(self, filename):
        # Current anesthetic IO seems cumbersome.
        raise NotImplementedError

    def read(self, filename):
        raise NotImplementedError


class NestedDiffusion(Integrator):
    def sample_constrained(self, n, dist, constraint, efficiency=0.5, **kwargs):
        success = []
        trials = 0
        while len(success) < n:
            batch_success = []
            pi = self.sample(n, dist, constraint, **kwargs)
            batch_success += [p for p in pi if p.logl > constraint]
            success += batch_success
            trials += n
            if len(batch_success) < n * efficiency:
                return False, len(success) / trials, batch_success
        return True, len(success) / trials, success

    def run(self, n=1000, target_eff=0.1, steps=20):
        live = self.sample(n * 10, self.prior, self.logzero)
        step = 0
        logger.info("Done sampling prior")
        live, contour = self.stash(live, n, drop=False)
        self.dist = self.prior
        self.update_stats(live, n)
        logger.info(f"{self.stats}")
        diffuser = self.model(self.prior)

        while step < steps:
            success, eff, points = self.sample_constrained(
                n,
                self.dist,
                contour,
                efficiency=target_eff,
            )
            step += 1
            live, contour = self.stash(live + points, n)

            if success:
                live, contour = self.stash(live + points, n)
                logger.info(f"Efficiency at: {eff}, using previous diffusion")
            self.update_stats(live, n)
            logger.info(f"{self.stats}")
            step += 1
            if not (success):
                logger.info(f"Efficiency dropped to: {eff}, training new diffusion")
                # diffuser = Diffusion(self.prior)
                diffuser.train(
                    np.asarray([yi.x for yi in live + points]),
                    n_epochs=len(live),
                    batch_size=n,
                    lr=1e-3,
                )
                live += points
                self.dists.append(diffuser)
                self.dist = diffuser
            logger.info(f"Step {step}/{steps} complete")

        self.stash(live, -len(live))
        logger.info(f"Final stats: {self.stats}")

    def update_stats(self, live, n):
        running_samples = self.points_to_samples(self.dead)
        self.stats.ndead = len(self.dead)
        lZs = running_samples.logZ(100)
        self.stats.logX = -np.asarray(
            [(x.logl_birth > self.logzero) for x in self.dead]
        ).sum() / (self.stats.nlive)
        self.stats.logz = lZs.mean()
        self.stats.logz_err = lZs.std()

    def points_to_samples(self, points):
        return NestedSamples(
            data=[p.x for p in points],
            logL=[p.logl for p in points],
            logL_birth=[p.logl_birth for p in points],
        )

    def points_to_samples_importance(self, points, weights):
        return MCMCSamples(
            data=[p.x for p in points],
            weights=weights,
            # weights=np.exp([p.logl_pi for p in points]),
        )

    def samples(self):
        return self.points_to_samples(self.dead)


class SequentialDiffusion(Integrator):
    beta_min = 0.00001
    beta_max = 1.0

    def run(self, n=1000, steps=10, schedule=np.linspace, **kwargs):
        target_eff = kwargs.get("efficiency", 0.5)

        betas = schedule(self.beta_min, self.beta_max, steps)
        self.dist = self.prior
        diffuser = self.model(self.prior)

        for beta_i in betas:
            live = self.sample(n * 2, self.dist, beta=beta_i)
            frame = self.points_to_samples(live)

            ess = len(frame.compress())
            while ess < n:
                live += self.sample(n, self.dist, beta=beta_i)
                frame = self.points_to_samples(live)
                ess = len(frame.compress())
                logger.info(f"Efficiency at: {ess/len(live)}, using previous diffusion")
            logger.info(f"Met ess criteria, training new diffusion")
            diffuser = self.model(self.prior)
            diffuser.train(
                np.asarray(self.points_to_samples(live).compress()),
                n_epochs=n,
                batch_size=n,
                lr=1e-3,
            )

            self.dist = diffuser
            self.dead += live
            self.update_stats(live, n)
            logger.info(f"{self.stats}")

        self.dead = self.sample(n * 4, self.dist)

    def update_stats(self, live, n):
        running_samples = self.points_to_samples(self.dead)
        self.stats.ndead = len(running_samples.compress())
        self.stats.logz = np.log(running_samples.get_weights().mean())

    def points_to_samples(self, points):
        if not points:
            return MCMCSamples(data=[], weights=[])
        else:
            logls = np.asarray([p.logl for p in points])
            logls -= logls.max()
            return MCMCSamples(data=[p.x for p in points], weights=np.exp(logls))
