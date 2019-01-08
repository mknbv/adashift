# Adashift

Reproducing of the [AdaShift: Decorrelation and Convergence of Adaptive Learning Rate Methods](https://openreview.net/forum?id=HkgTkhRcKQ)
as a part of ICLR Reproducibility Challange 2019. See our [report](https://github.com/MichaelKonobeev/adashift/tree/master/report/report.pdf).


## Experiments

**Synthetic Experiment**

![Synthetic](/img/regret_synth.png) ![Synthetic_optval](/img/opt_synth.png)


**Logistic Regression on MNIST**

![LR1](/img/mnist_LR_smooth_1000.png) ![LR2](/img/mnist_LR_1000.png)

**W-GAN**

![wgan-loss](/img/wgan-discriminator-loss.png) ![wgan-inception-score](/img/wgan-is.png)

![fixed-gen](/img/discriminator_fixed_gen.png)

**NMT**

![nmt](https://raw.githubusercontent.com/MichaelKonobeev/adashift/master/img/nmt_adam.png)

## Dependencies

- Python 3.4+,
- NumPy, Pandas
- PyTorch 0.4+
- SciPy, Matplotlib
- OpenNMT-py for NMT
- GPU for DL experiments
### References

[On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ&utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=piqcy)

[ Adam ](https://arxiv.org/abs/1412.6980)


[AdaShift: Decorrelation and Convergence of Adaptive Learning Rate Methods](https://arxiv.org/abs/1810.00143)
