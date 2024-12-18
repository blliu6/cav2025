include("utils/util.jl")
include("utils/newtonUpdate.jl")
include("utils/rationalSOS.jl")

X = nothing
eig_tol = 1e-4
tol = 1e-50


function solve(n, X, f, Q, mono,newton)
    
    Q = round.(Q, digits=2)
    rQ = symplify_rational.(Q)

    model = nothing
    if !newton
        model = rational_SOS(f, 1, 0, mono, rQ)
    else
        update_Q = newton_refine_update(f, mono, Q, eig_tol, 90, tol, X)
        Q_res = symplify_rational.(update_Q*update_Q')
        model = rational_SOS(f, 1, 0, mono, Q_res)
    end
    
    SOSrational_time, coeffs, terms = model[2:end]
    
    @show expand(f-(coeffs' * terms.^2))

    data = Dict(
        "n" => n,
        "expression" => f,
        "coeff" => join([string(c) for c in coeffs], ","),
        "term" => join([string(t) for t in terms], ","),
        "SOSrational_time" => SOSrational_time,
        "SOS_expression" => string(coeffs' * terms.^2)
    )

    # save("src/output/data.json", data)    
end

n = 2
@Symbolics.variables X[1:n]
#C5
# init
## Multiplier-2
P0 = 1469*X[1]^2//250 - 1071*X[1]*X[2]//5000 + 2153*X[1]//1000 + 6173*X[2]^2//1000 - 3023*X[2]//10000 + 756//125
M0 = [1, X[2], X[1]]
Q0 = [ 6.05e+00 -1.54e-01  1.08e+00;
-1.54e-01  6.17e+00 -1.09e-01;
 1.08e+00 -1.09e-01  5.88e+00;]
solve(n, X, P0, Q0, M0, true)
## Total Decomposition
P1 = 1819*X[1]^4//500 - 3481*X[1]^3*X[2]//12500 - 18067*X[1]^3//2000 + 11781*X[1]^2*X[2]^2//1250 + 35519*X[1]^2*X[2]//100000 + 16729*X[1]^2//2500 - 447*X[1]*X[2]^3//2000 - 84657*X[1]*X[2]^2//10000 + 39687*X[1]*X[2]//5000 + 4607*X[1]//2000 + 15439*X[2]^4//2500 - 7831*X[2]^3//25000 + 7591*X[2]^2//1000 + 5551*X[2]//500 + 8593//500
M1 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q1 = [ 1.72e+01  5.55e+00  1.16e+00 -9.55e-01  2.26e+00 -1.97e+00;
 5.55e+00  9.52e+00  1.70e+00 -1.54e-01 -1.66e+00 -7.64e-01;
 1.16e+00  1.70e+00  1.06e+01 -2.57e+00  9.38e-01 -4.52e+00;
-9.55e-01 -1.54e-01 -2.57e+00  6.17e+00 -1.09e-01  1.06e+00;
 2.26e+00 -1.66e+00  9.38e-01 -1.09e-01  7.29e+00 -1.39e-01;
-1.97e+00 -7.64e-01 -4.52e+00  1.06e+00 -1.39e-01  3.64e+00;]
solve(n, X, P1, Q1, M1, true)
## Multiplier-1
P2 = 91*X[1]^2//25 - 2751*X[1]*X[2]//10000 + 939*X[1]//500 + 1769*X[2]^2//500 - 4843*X[2]//10000 + 1893//500
M2 = [1, X[2], X[1]]
Q2 = [ 3.79e+00 -2.44e-01  9.40e-01;
-2.44e-01  3.54e+00 -1.39e-01;
 9.40e-01 -1.39e-01  3.64e+00;]
solve(n, X, P2, Q2, M2, true)
# unsafe
## Multiplier-2
P3 = 5417*X[1]^2//2000 - 11629*X[1]*X[2]//5000 + 37561*X[1]//10000 + 4423*X[2]^2//1000 - 66611*X[2]//10000 + 1297//125
M3 = [1, X[2], X[1]]
Q3 = [ 1.04e+01 -3.33e+00  1.88e+00;
-3.33e+00  4.42e+00 -1.16e+00;
 1.88e+00 -1.16e+00  2.71e+00;]
solve(n, X, P3, Q3, M3, true)
## Total Decomposition
P4 = 17*X[1]^4//8 - 1797*X[1]^3*X[2]//1000 + 2651*X[1]^3//500 + 5649*X[1]^2*X[2]^2//1000 + 1463*X[1]^2*X[2]//500 + 2123*X[1]^2//250 - 2329*X[1]*X[2]^3//1000 + 4983*X[1]*X[2]^2//1000 - 3633*X[1]*X[2]//5000 + 464*X[1]//125 + 2211*X[2]^4//500 + 109*X[2]^3//50 + 1803*X[2]^2//250 + 4071*X[2]//1000 + 921//500
M4 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q4 = [ 1.84e+00  2.03e+00  1.86e+00  6.84e-01  3.82e-01  7.33e-01;
 2.03e+00  5.84e+00 -7.47e-01  1.09e+00  8.10e-01 -2.56e-02;
 1.86e+00 -7.47e-01  7.03e+00  1.68e+00  1.49e+00  2.65e+00;
 6.84e-01  1.09e+00  1.68e+00  4.42e+00 -1.16e+00  5.22e-01;
 3.82e-01  8.10e-01  1.49e+00 -1.16e+00  4.60e+00 -8.96e-01;
 7.33e-01 -2.56e-02  2.65e+00  5.22e-01 -8.96e-01  2.13e+00;]
solve(n, X, P4, Q4, M4, true)
## Multiplier-1
P5 = 1063*X[1]^2//500 - 224*X[1]*X[2]//125 + 1053*X[1]//1000 + 1469*X[2]^2//500 + 547*X[2]//500 + 5063//1000
M5 = [1, X[2], X[1]]
Q5 = [ 5.06e+00  5.47e-01  5.27e-01;
 5.47e-01  2.94e+00 -8.96e-01;
 5.27e-01 -8.96e-01  2.13e+00;]
solve(n, X, P5, Q5, M5, true)
# Lie
## Multiplier-2
P6 = 13*X[1]^2//100 + 57*X[1]*X[2]//125 + X[1]//4 + 751*X[2]^2//1000 + 831*X[2]//1000 + 341//1000
M6 = [1, X[2], X[1]]
Q6 = [ 3.41e-01  4.16e-01  1.25e-01;
 4.16e-01  7.52e-01  2.28e-01;
 1.25e-01  2.28e-01  1.30e-01;]
solve(n, X, P6, Q6, M6, true)
## Total Decomposition
P7 = 1197*X[1]^4//500 - 2339*X[1]^3*X[2]//1000 + 909*X[1]^3//250 + 6411*X[1]^2*X[2]^2//10000 - 4877*X[1]^2*X[2]//5000 - 273*X[1]^2//40 + 229*X[1]*X[2]^3//500 + 1221*X[1]*X[2]^2//1000 + 1783*X[1]*X[2]//250 - 5373*X[1]//1000 + 1883*X[2]^4//2500 + 2319*X[2]^3//1000 + 4521*X[2]^2//1000 + 2813*X[2]//1000 + 363//40
M7 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q7 = [ 9.07e+00  1.41e+00 -2.69e+00  7.41e-01  2.67e+00 -4.34e+00;
 1.41e+00  3.03e+00  8.89e-01  1.16e+00  5.58e-01  2.84e-02;
-2.69e+00  8.89e-01  1.85e+00  5.31e-02 -5.15e-01  1.82e+00;
 7.41e-01  1.16e+00  5.31e-02  7.52e-01  2.28e-01 -2.25e-01;
 2.67e+00  5.58e-01 -5.15e-01  2.28e-01  1.09e+00 -1.17e+00;
-4.34e+00  2.84e-02  1.82e+00 -2.25e-01 -1.17e+00  2.39e+00;]
solve(n, X, P7, Q7, M7, true)
## Multiplier-1
P8 = 703*X[1]^2//10000 + 63*X[1]*X[2]//200 + 239*X[1]//1000 + 64*X[2]^2//125 + 147*X[2]//200 + 313//1000
M8 = [1, X[2], X[1]]
Q8 = [ 3.13e-01  3.68e-01  1.19e-01;
 3.68e-01  5.13e-01  1.58e-01;
 1.19e-01  1.58e-01  7.01e-02;]
solve(n, X, P8, Q8, M8, true)