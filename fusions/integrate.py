import logging

from fusions.cfm import CFM
from fusions.diffusion import Diffusion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pickle import dump, load

import anesthetic.termination as term
import matplotlib.pyplot as plt
import numpy as np
from anesthetic import MCMCSamples, NestedSamples, make_2d_axes, read_chains
from anesthetic.utils import compress_weights, neff

# from anesthetic.read.hdf import read_hdf, write_hdf
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from tqdm import tqdm

from fusions.model import Model
from fusions.utils import unit_hyperball, unit_hypercube
from jax import random


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


@dataclass
class Settings:
    n: int = 500
    target_eff: float = 0.1
    steps: int = 20
    prior_boost: int = 5
    eps: float = 1e-3
    batch_size: float = 0.25
    epoch_factor: int = 1
    restart: bool = False
    noise: float = 1e-3
    # efficiency: float = 1 / np.e
    # logzero: float = -1e30

    def __repr__(self):
        return (
            f"Settings(\n"
            f"  n: {self.n},\n"
            f"  target_eff: {self.target_eff},\n"
            f"  steps: {self.steps},\n"
            f"  prior_boost: {self.prior_boost},\n"
            f"  eps: {self.eps},\n"
            f"  batch_size: {self.batch_size},\n"
            f"  epoch_factor: {self.epoch_factor},\n"
            f"  restart: {self.restart},\n"
            f"  noise: {self.noise},\n"
            # f"  efficiency: {self.efficiency},\n"
            f")"
        )


@dataclass
class Trace:
    diff: Point = field(default_factory=dict)
    live: Point = field(default_factory=dict)
    accepted_live: Point = field(default_factory=dict)
    iteration: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=dict)


class Integrator(ABC):
    def __init__(self, prior, likelihood, **kwargs) -> None:
        self.prior = prior
        self.likelihood = likelihood
        self.logzero = kwargs.get("logzero", -np.inf)
        self.dead = []
        self.dists = []
        self.stats = Stats()
        self.settings = Settings()
        self.model = kwargs.get("model", CFM)
        latent = kwargs.get("latent", unit_hyperball)
        # latent = multivariate_normal(mean=np.zeros(prior.dim), cov=np.eye(prior.dim))
        self.latent = latent(prior.dim, scale=1.0)
        self.dim = prior.dim
        self.rng = kwargs.get("rng", random.PRNGKey(0))
        self.trace = Trace()
        # self.latent = multivariate_normal(
        #     np.zeros(prior.dim), np.eye(prior.dim)
        # )

    def sample(self, n, dist, logl_birth=0.0, beta=1.0):
        if isinstance(dist, Model):
            x, j = dist.rvs(n, jac=True, solution="none")
            x = np.asarray(x)
            w = np.ones(n)
        else:
            x = np.asarray(dist.rvs(n))
            w = np.ones(n)
        log_pi = self.prior.logpdf(x)

        idx = compress_weights(w.flatten(), ncompress="equal")
        idx = np.asarray(idx, dtype=bool)

        logl = self.likelihood.logpdf(x) * beta
        self.stats.nlike += idx.sum()
        logl_birth = np.ones_like(logl) * logl_birth
        points = [
            Point(
                x[idx][i],
                logl[idx][i],
                logl_birth[idx][i],
                logl_pi=log_pi[idx][i],
            )
            for i in range(idx.sum())
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
        self.points_to_samples(self.dead).to_csv(filename + ".csv")

    def read(self, filename):
        self.dead = read_chains(filename)


class NestedDiffusion(Integrator):
    def sample_constrained(self, n, dist, constraint, efficiency=0.1, **kwargs):
        success = []
        trials = 0
        while len(success) < n:
            batch_success = []
            pi = self.sample(n, dist, constraint, **kwargs)
            batch_success += [p for p in pi if p.logl > constraint]
            success += batch_success
            trials += n
        eff = len(success) / trials
        return eff > efficiency, eff, success

    def train_diffuser(self, dist, points):
        dist.train(
            np.asarray([yi.x for yi in points]),
            n_epochs=int(len(points) * self.settings.epoch_factor),
            batch_size=int(len(points) * self.settings.batch_size),
            lr=1e-3,
            restart=self.settings.restart,
            noise=self.settings.noise,
        )
        return dist

    def run(self):
        print(self.settings)
        n = self.settings.n
        live = self.sample(n * self.settings.prior_boost, self.prior, self.logzero)

        step = 0
        logger.info("Done sampling prior")
        live, contour = self.stash(live, n // 2, drop=False)
        self.dist = self.prior
        self.update_stats(live, n)
        logger.info(f"{self.stats}")
        diffuser = self.model(self.latent, noise=1e-3)
        diffuser.rng, _ = random.split(diffuser.rng)

        while not self.points_to_samples(live + self.dead).terminated(
            "logZ", self.settings.eps
        ):
            success, eff, points = self.sample_constrained(
                n // 2,
                self.dist,
                contour,
                efficiency=self.settings.target_eff,
            )
            self.trace.live[step] = live
            live = live + points
            self.trace.accepted_live[step] = points
            live, contour = self.stash(live, n // 2)
            # x = self.dist.rvs(len(live))
            # self.trace.diff[step] = x
            # t = np.asarray([yi.x for yi in live])
            # self.dist.calibrate(np.asarray([yi.x for yi in live]), np.asarray(x), n_epochs=len(live) * 5, batch_size=n, restart =True)

            if success:
                logger.info(f"Efficiency at: {eff}, using previous diffusion")
            if not success:
                logger.info(f"Efficiency dropped to: {eff}, training new diffusion")
                diffuser = self.train_diffuser(diffuser, points)
                self.trace.losses[step] = diffuser.trace.losses
                self.dists.append(diffuser)
                self.dist = diffuser

            self.update_stats(live, n)
            logger.info(f"{self.stats}")
            step += 1
            self.trace.iteration.append(step)

            logger.info(f"Step {step} complete")

        self.stash(live, -len(live))
        dump(self.trace, open("plots/trace.pkl", "wb"))
        logger.info(f"Final stats: {self.stats}")

    def update_stats(self, live, n):
        running_samples = self.points_to_samples(self.dead + live)
        self.stats.ndead = len(self.dead)
        lZs = running_samples.logZ(100)
        # self.stats.logX = running_samples.critical_ratio()
        running_samples.terminated()
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
