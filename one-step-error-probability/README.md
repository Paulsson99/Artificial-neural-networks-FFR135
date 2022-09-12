We want to estimate the one step error probability $P_\mathrm{Error}^{t=1}$ for an asynchrons update

$$s_m' = \mathrm{sgn}\left(\sum_i w_{mi}s_i\right),$$

when storing $p$ random patterns with $N$ bits in the weights and feeding the network one of those patterns, e.g.

$$
w_{ij} = \frac{1}{N}\sum_\mu x_i^\mu x_j^\mu \quad (w_{ii}=0), \\
s_i = x_i^\nu \quad \nu\in[1, 2, \dots, \mu].
$$

This can be done by generating $M$ random samples of weights and do the calculations and count how many times, $m$, $s_m'\ne s_m$. $P_\mathrm{Error}^{t=1}$ can then be estimated as

$$P_\mathrm{Error}^{t=1}\approx\frac{m}{M}.$$

My first atempt to solve this involved generating all the random patterns and calculating the full weight matrix. But this takes a loooong time ($\sim 1\mathrm{h}$). So to improve the calculation I made use of the **Cross-Talk Term** $C_i^\nu$.

The cross-talk term can be derived by expanding the right hand side of the first equation. 

$$
\mathrm{sgn}\left(\sum_i w_{mi}s_i\right) = 
\mathrm{sgn}\left(\sum_i w_{mi}x_i^\nu\right) = 
\mathrm{sgn}\left(\frac{1}{N}\sum_i \sum_\mu x_m^\mu x_i^\mu x_i^\nu\right) =
\mathrm{sgn}\left(b_m^\nu\right),
$$
with
$$
b_m^\nu = \frac{1}{N}\sum_i \sum_\mu x_m^\mu x_i^\mu x_i^\nu = 
x_i^\nu + \frac{1}{N}\sum_i \sum_{\mu\ne\nu} x_m^\mu x_i^\mu x_i^\nu.
$$
If $w_{ii}=0$ however we instead get
$$
b_m^\nu = \left(1-\frac{1}{N}\right)x_i^\nu + \frac{1}{N}\sum_{i\ne m} \sum_{\mu\ne\nu} x_m^\mu x_i^\mu x_i^\nu.
$$

Now we define the cross-talk term as

$$
C_m^\nu = -x_m^\nu\frac{1}{N}\sum_i \sum_{\mu\ne\nu} x_m^\mu x_i^\mu x_i^\nu =
-\frac{1}{N}\sum_i \sum_{\mu\ne\nu} x_m^\mu x_i^\mu x_i^\nu x_m^\nu
\quad (\mathrm{or}\; i\ne m \; \mathrm{if} \; w_{ii} = 0)
$$

Now for all $i\ne m$, the numbers $x_m^\mu, x_i^\mu, x_i^\nu, x_m^\nu$ are independent random numbers that take the value of $+1$ or $-1$. And for $i=m$ we get $x_m^\mu x_m^\mu x_m^\nu x_m^\nu = 1$. So 
$$
C_m^\nu = -\frac{1}{N}\sum_{\mu\ne\nu}\left(\sum_{i\ne m}r_i + 1\right) = 
-\frac{1}{N}\left(\sum_{i}^{K}r_i + p-1\right),
$$
with $K=(N-1)(p-1)$ and $r_i$ is $+1$ or $-1$ with equal probability. If we have $w_{ii}=0$ the last term $p-1$ would vanish. 

The sum $\sum_{i}^{K}r_i = P_B - (K - P_B) = 2P_B - K$, where $P_B$ is sampled from a binomial distrubution. So 
$$
C_m^\nu = -\frac{2P_B - K + p - 1}{N}, \\
\left(C_m^\nu = -\frac{2P_B - K}{N} \quad \mathrm{if}\;w_{ii}=0\right).
$$
The simplification comes when we realise that there is only an error if $C_m^\nu>1$. So to estimate $P_\mathrm{Error}^{t=1}$ we only need to draw $M$ samples from a binomial distrubution and count how often the quantity $C_m^\nu>1$ ($C_m^\nu>1-\frac{1}{N}$ for $w_{ii}=0$).

With this method the computation time is reduce from the order of one hour to one second. An improvment of roughly 3600 times. 