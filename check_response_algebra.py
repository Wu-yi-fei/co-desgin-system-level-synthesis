"""Exact rational checks for the manuscript's finite horizon response identities."""
from fractions import Fraction as F
import itertools
import random


checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)


def zeros(r, c):
    return [[F(0) for _ in range(c)] for _ in range(r)]


def eye(n):
    a = zeros(n, n)
    for i in range(n):
        a[i][i] = F(1)
    return a


def add(a, b, sign=1):
    return [[x + sign * y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def mul(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), F(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def inv(a):
    n = len(a)
    v = [row[:] + unit for row, unit in zip(a, eye(n))]
    for k in range(n):
        pivot = next(i for i in range(k, n) if v[i][k])
        v[k], v[pivot] = v[pivot], v[k]
        divisor = v[k][k]
        v[k] = [x / divisor for x in v[k]]
        for i in range(n):
            if i != k:
                multiplier = v[i][k]
                v[i] = [x - multiplier * y for x, y in zip(v[i], v[k])]
    return [row[n:] for row in v]


def random_matrix(r, c, rng):
    return [[F(rng.randint(-3, 3), rng.randint(1, 5)) for _ in range(c)]
            for _ in range(r)]


def mask_input(a, T, m, col_size, releases):
    for t in range(T):
        for s, release in enumerate(releases):
            if t < release:
                for i in range(m):
                    for j in range(col_size):
                        a[t*m+i][s*col_size+j] = F(0)
    return a


def check_masks(px, pu, sx, su, T, n, m, d, releases):
    for t in range(T+1):
        for s in range(T+1):
            if t <= s:
                check(all(px[t*n+i][s*n+j] == (F(i == j) if t == s else 0)
                          for i in range(n) for j in range(n)), 'Phi_x timing')
            if t < T and t < s:
                check(all(pu[t*m+i][s*n+j] == 0 for i in range(m) for j in range(n)),
                      'Phi_u timing')
            if t <= releases[s]:
                check(all(sx[t*n+i][s*d+j] == 0 for i in range(n) for j in range(d)),
                      'Psi_x release and plant delay')
            if t < T and t < releases[s]:
                check(all(su[t*m+i][s*d+j] == 0 for i in range(m) for j in range(d)),
                      'Psi_u release')


forecast_suites = 0
ordinary_suites = 0
for T, (n, m) in itertools.product((1, 3), ((2, 1), (1, 2), (3, 2))):
    rng = random.Random(100*T + 10*n + m)
    nx, nu, d = (T+1)*n, T*m, 2
    D, B = eye(nx), zeros(nx, nu)
    A = [random_matrix(n, n, rng) for _ in range(T)]
    plant_B = [random_matrix(n, m, rng) for _ in range(T)]
    for s in range(T):
        for i in range(n):
            for j in range(n):
                D[(s+1)*n+i][s*n+j] = -A[s][i][j]
            for j in range(m):
                B[(s+1)*n+i][s*m+j] = plant_B[s][i][j]
    Dinv = inv(D)
    K = mask_input(random_matrix(nu, nx, rng), T, m, n, range(T+1))
    px = inv(add(D, mul(B, K), -1))
    pu = mul(K, px)
    check(add(mul(D, px), mul(B, pu), -1) == eye(nx), 'ordinary SLS equation')
    check(mul(pu, inv(px)) == K, 'ordinary policy recovery')
    check(len(pu) == T*m and len(px) == (T+1)*n, 'terminal dimensions')
    ordinary_suites += 1
    for releases in ([0]*(T+1), list(range(T+1)), [s//2 for s in range(T+1)]):
        np = (T+1)*d
        L = mask_input(random_matrix(nu, np, rng), T, m, d, releases)
        sx = mul(mul(px, B), L)
        su = add(mul(K, sx), L)
        check(add(mul(D, sx), mul(B, su), -1) == zeros(nx, np), 'forecast SLS equation')
        check(add(su, mul(K, sx), -1) == L, 'forecast policy recovery')
        check_masks(px, pu, sx, su, T, n, m, d, releases)
        eta, p = random_matrix(nx, 1, rng), random_matrix(np, 1, rng)
        x = add(mul(px, eta), mul(sx, p))
        u = add(mul(pu, eta), mul(su, p))
        check(add(mul(D, x), mul(B, u), -1) == eta, 'plant trajectory')
        check(add(mul(K, x), mul(L, p)) == u, 'policy trajectory')
        forecast_suites += 1

        # Independently choose feasible response coordinates, then reconstruct policy.
        pu2 = mask_input(random_matrix(nu, nx, rng), T, m, n, range(T+1))
        su2 = mask_input(random_matrix(nu, np, rng), T, m, d, releases)
        px2 = mul(Dinv, add(eye(nx), mul(B, pu2)))
        sx2 = mul(mul(Dinv, B), su2)
        K2 = mul(pu2, inv(px2))
        L2 = add(su2, mul(K2, sx2), -1)
        check(px2 == inv(add(D, mul(B, K2), -1)), 'reverse Phi_x')
        check(pu2 == mul(K2, px2), 'reverse Phi_u')
        check(sx2 == mul(mul(px2, B), L2), 'reverse Psi_x')
        check(su2 == add(mul(K2, sx2), L2), 'reverse Psi_u')
        check_masks(px2, pu2, sx2, su2, T, n, m, d, releases)
        check(mask_input([row[:] for row in K2], T, m, n, range(T+1)) == K2,
              'recovered K causality')
        check(mask_input([row[:] for row in L2], T, m, d, releases) == L2,
              'recovered L releases')
        forecast_suites += 1

# Observation models differ even without forecasts and for a stable plant.
# The empty communication graph permits diagonal U but reconstruction also
# requires the off-diagonal blocks of X to vanish.
a = [[F(0), F(1, 3)], [F(0), F(0)]]
d = eye(6)
b = zeros(6, 4)
for t in range(2):
    for i in range(2):
        for j in range(2):
            d[2*(t+1)+i][2*t+j] = -a[i][j]
            b[2*(t+1)+i][2*t+j] = F(i == j)
u = zeros(4, 6)
x = inv(d)
check(add(mul(d, x), mul(b, u), -1) == eye(6),
      'observed innovation zero controller satisfies the full SLS equation')
check(x[2][1] == F(1, 3),
      'physical propagation persists without communication')
check(mul(a, a) == zeros(2, 2), 'strict inclusion example is nilpotent')
for diagonal in itertools.product((F(-1), F(0), F(1)), repeat=2):
    u00 = [[diagonal[0], F(0)], [F(0), diagonal[1]]]
    check(add(a, u00)[0][1] == F(1, 3),
          'diagonal first input response cannot cancel the off-diagonal state response')

print(f'PASS {checks} exact rational assertions')
print(f'PASS {forecast_suites} forecast suites and {ordinary_suites} ordinary SLS suites')
print('T in {1,3}; (n,m) in {(2,1),(1,2),(3,2)}; zero, target-time, mixed releases')
print('Both response-to-policy and policy-to-response maps checked with terminal state and no terminal input')
print('Observed and reconstructed innovation models checked on the strict inclusion example')
