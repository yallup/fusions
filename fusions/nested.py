import logging

from fusions.cfm import CFM
from fusions.diffusion import Diffusion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
from dataclasses import dataclass, field

import jax
import matplotlib.pyplot as plt
import numpy as np
from anesthetic import NestedSamples
from scipy.special import logsumexp
from tqdm import tqdm

os.makedirs("plots", exist_ok=True)


@dataclass
class Point:
    x: np.ndarray
    logl: float
    logl_birth: float


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


def plot_points(points):
    plt.scatter(*np.asarray([p.x for p in points]).T, c=[p.logl for p in points])
    # plt.show()


class NestedDiffusion(object):
    def __init__(self, prior, likelihood) -> None:
        self.prior = prior
        self.likelihood = likelihood
        self.ndims = None
        # self.rng = random.PRNGKey(2022)
        self.state = None
        self.sigma = 0.1
        self.logzero = -1e30
        self.dead = []
        self.dists = []
        self.stats = Stats()
        self.last_live = None

    def sample(self, n, dist, logl_birth):
        x = np.asarray(dist.rvs(n))
        logl = self.likelihood.logpdf(x)
        self.stats.nlike += n
        logl_birth = np.ones_like(logl) * logl_birth
        points = [Point(x[i], logl[i], logl_birth[i]) for i in range(x.shape[0])]
        return points

    def sample_constrained(self, n, dist, constraint, efficiency=0.1):
        success = []
        trials = 0
        while len(success) < n:
            batch_success = []
            pi = self.sample(n, dist, constraint)
            batch_success += [p for p in pi if p.logl > constraint]
            success += batch_success
            trials += n
            if len(batch_success) < n * efficiency:
                return False, len(success) / trials, batch_success
        return True, len(success) / trials, success

    def stash(self, points, n):
        live = sorted(points, key=lambda lp: lp.logl)

        self.dead += live[:-n]
        self.last_live = live[:-n]
        contour = live[n].logl
        live = live[-n:]
        return live, contour

    def run(self, n=1000, target_eff=0.1, steps=None, eps=-3):
        self.stats.nlive = n
        live = self.sample(n * 2, self.prior, self.logzero)
        step = 0
        logger.info("Done sampling prior")
        live, contour = self.stash(live, n)

        while step < steps:
            # if self.stats.logX<-3:
            #     self.stash(live, -len(live))
            #     logger.info("Precision reached, terminating")
            #     break
            self.update_stats(live, n)
            logger.info(f"\r{self.stats}")
            success, eff, points = self.sample_constrained(
                n, self.prior, contour, efficiency=target_eff
            )
            if success:
                live, contour = self.stash(live + points, n)
                step += 1
                logger.info(f"\rEfficiency at: {eff}, using previous diffusion")

            if not (success):
                logger.info(f"\rEfficiency dropped to: {eff}, training new diffusion")
                # diffuser = CFM(self.prior)
                diffuser = Diffusion(self.prior)
                # with jax.disable_jit():
                diffuser.train(
                    np.asarray([yi.x for yi in live + points]),
                    n_epochs=500,
                    batch_size=n // 2,
                    lr=1e-3,
                )
                live += points
                # live, contour = self.stash(live + points, n)

                self.dists.append(self.prior)
                self.prior = diffuser

            logger.info(f"Step {step}/{steps} complete")

        # mean_logl = np.mean([p.logl for p in live])
        # average_live = lambda x: x.logl = mean_logl
        # live = [average_live(p) for p in live]
        self.stash(live, -len(live))
        print("done")

    def update_stats(self, live, n):
        running_samples = self.points_to_samples(self.dead + live)
        live_samples = self.points_to_samples(live)
        self.stats.ndead = len(self.dead)
        lZs = running_samples.logZ(100)
        # self.stats.logX = -np.asarray(
        #     [(x.logl_birth > self.logzero) for x in self.dead]
        # ).sum() / (self.stats.nlive)
        # self.stats.logX = live_samples.logZ() - running_samples.logZ()
        self.stats.logX = logsumexp(live_samples.logX() * live_samples.logL)
        self.stats.logz = lZs.mean()
        self.stats.logz_err = lZs.std()

    def points_to_samples(self, points):
        return NestedSamples(
            data=[p.x for p in points],
            logL=[p.logl for p in points],
            logL_birth=[p.logl_birth for p in points],
        )

    def samples(self):
        return self.points_to_samples(self.dead)
