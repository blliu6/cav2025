include("utils/util.jl")
include("utils/newtonUpdate.jl")
include("utils/rationalSOS.jl")

X = nothing
eig_tol = 1e-4
tol = 1e-50


SOS_time = 0
Newton_time = 0
function solve(n, X, f, Q, mono,newton)
    global SOS_time, Newton_time
    Q = round.(Q, digits=2)
    rQ = symplify_rational.(Q)

    model = nothing
    if !newton
        SOS_time+=@elapsed model = rational_SOS(f, 1, 0, mono, rQ)
    else
        Newton_time+=@elapsed update_Q = newton_refine_update(f, mono, Q, eig_tol, 90, tol, X)
        Q_res = symplify_rational.(update_Q*update_Q')
        SOS_time+=@elapsed model = rational_SOS(f, 1, 0, mono, Q_res)
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

# init
## Multiplier-2
P0 = 2813*X[1]^2//500 - 96*X[1]*X[2]//625 - 3119*X[1]//1000 + 53*X[2]^2//10 + 963*X[2]//500 + 339//50
M0 = [1, X[2], X[1]]
Q0 = [ 6.78e+00  9.66e-01 -1.56e+00;
 9.66e-01  5.30e+00 -7.55e-02;
-1.56e+00 -7.55e-02  5.63e+00;]

rational_SOS(P0, 1, 0, M0, Q0)

solve(n, X, P0, Q0, M0, false)
## Total Decomposition
P1 = 22*X[1]^4//5 + 1743*X[1]^3*X[2]//5000 + 3349*X[1]^3//500 + 1899*X[1]^2*X[2]^2//200 - 1311*X[1]^2*X[2]//250 + 2591*X[1]^2//500 - 1503*X[1]*X[2]^3//10000 + 4749*X[1]*X[2]^2//1000 + 2781*X[1]*X[2]//1000 + 1813*X[1]//200 + 5301*X[2]^4//1000 - 843*X[2]^3//250 + 4829*X[2]^2//500 + 2081*X[2]//250 + 2209//250
M1 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q1 = [ 8.84e+00  4.17e+00  4.53e+00 -1.65e-01 -4.96e-01 -1.68e+00;
 4.17e+00  1.00e+01  1.89e+00 -1.68e+00  1.21e+00 -1.44e+00;
 4.53e+00  1.89e+00  8.54e+00  1.16e+00 -1.18e+00  3.35e+00;
-1.65e-01 -1.68e+00  1.16e+00  5.30e+00 -7.55e-02  1.04e+00;
-4.96e-01  1.21e+00 -1.18e+00 -7.55e-02  7.40e+00  1.79e-01;
-1.68e+00 -1.44e+00  3.35e+00  1.04e+00  1.79e-01  4.40e+00;]
solve(n, X, P1, Q1, M1, false)
## Multiplier-1
P2 = 2199*X[1]^2//500 + 3557*X[1]*X[2]//10000 - 2099*X[1]//1000 + 3861*X[2]^2//1000 - 1631*X[2]//5000 + 497//125
M2 = [1, X[2], X[1]]
Q2 = [ 3.98e+00 -1.63e-01 -1.05e+00;
-1.63e-01  3.86e+00  1.79e-01;
-1.05e+00  1.79e-01  4.40e+00;]
solve(n, X, P2, Q2, M2, false)
# unsafe
## Multiplier-2
P3 = 1091*X[1]^2//200 - 7*X[1]*X[2]//40 - 3191*X[1]//1000 + 5653*X[2]^2//1000 - 1073*X[2]//500 + 2333//250
M3 = [1, X[2], X[1]]
Q3 = [ 9.33e+00 -1.07e+00 -1.60e+00;
-1.07e+00  5.65e+00 -8.61e-02;
-1.60e+00 -8.61e-02  5.46e+00;]
solve(n, X, P3, Q3, M3, false)
## Total Decomposition
P4 = 24617*X[1]^4//5000 - 79361*X[1]^3*X[2]//100000 + 657*X[1]^3//100 + 96987*X[1]^2*X[2]^2//10000 + 51019*X[1]^2*X[2]//10000 + 1137*X[1]^2//200 - 17519*X[1]*X[2]^3//100000 + 51269*X[1]*X[2]^2//10000 - 15181*X[1]*X[2]//10000 + 5139*X[1]//1000 + 3533*X[2]^4//625 + 35019*X[2]^3//10000 + 6223*X[2]^2//500 - 4951*X[2]//1000 + 4477//1000
M4 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q4 = [ 4.48e+00 -2.48e+00  2.57e+00  2.78e-01 -5.53e-02 -8.15e-01;
-2.48e+00  1.19e+01 -7.05e-01  1.75e+00  1.54e+00  1.82e+00;
 2.57e+00 -7.05e-01  7.31e+00  1.02e+00  7.31e-01  3.29e+00;
 2.78e-01  1.75e+00  1.02e+00  5.65e+00 -8.61e-02  8.93e-01;
-5.53e-02  1.54e+00  7.31e-01 -8.61e-02  7.91e+00 -3.99e-01;
-8.15e-01  1.82e+00  3.29e+00  8.93e-01 -3.99e-01  4.93e+00;]
solve(n, X, P4, Q4, M4, false)
## Multiplier-1
P5 = 4927*X[1]^2//1000 - 199*X[1]*X[2]//250 - 657*X[1]//200 + 849*X[2]^2//200 + 309*X[2]//250 + 6367//1000
M5 = [1, X[2], X[1]]
Q5 = [ 6.37e+00  6.18e-01 -1.64e+00;
 6.18e-01  4.24e+00 -3.99e-01;
-1.64e+00 -3.99e-01  4.93e+00;]
solve(n, X, P5, Q5, M5, false)
# Lie
## Multiplier-2
P6 = 699*X[1]^2//500 - 113*X[1]*X[2]//500 + 247*X[1]//2500 + 101*X[2]^2//100 + 7*X[2]//25 + 87//500
M6 = [1, X[2], X[1]]
Q6 = [ 1.74e-01  1.40e-01  4.93e-02;
 1.40e-01  1.01e+00 -1.13e-01;
 4.93e-02 -1.13e-01  1.40e+00;]
solve(n, X, P6, Q6, M6, false)
## Total Decomposition
P7 = 831*X[1]^4//500 - 2663*X[1]^3*X[2]//10000 + 231*X[1]^3//2000 + 437*X[1]^2*X[2]^2//200 + 2401*X[1]^2*X[2]//10000 + 1861*X[1]^2//1000 - 142*X[1]*X[2]^3//625 + 2471*X[1]*X[2]^2//25000 - 429*X[1]*X[2]//2000 + 117*X[1]//1000 + 1007*X[2]^4//1000 + 2807*X[2]^3//10000 + 13*X[2]^2//10 + 157*X[2]//500 + 581//1000
M7 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q7 = [ 5.81e-01  1.57e-01  5.87e-02  1.60e-02 -4.36e-02  1.43e-01;
 1.57e-01  1.27e+00 -6.43e-02  1.40e-01  7.04e-03  1.43e-01;
 5.87e-02 -6.43e-02  1.57e+00  4.22e-02 -2.33e-02  5.84e-02;
 1.60e-02  1.40e-01  4.22e-02  1.01e+00 -1.13e-01  3.36e-01;
-4.36e-02  7.04e-03 -2.33e-02 -1.13e-01  1.51e+00 -1.33e-01;
 1.43e-01  1.43e-01  5.84e-02  3.36e-01 -1.33e-01  1.66e+00;]
solve(n, X, P7, Q7, M7, false)
## Multiplier-1
P8 = 831*X[1]^2//500 - 133*X[1]*X[2]//500 + 117*X[1]//1000 + 197*X[2]^2//250 + 6*X[2]//25 + 79//500
M8 = [1, X[2], X[1]]
Q8 = [ 1.58e-01  1.20e-01  5.84e-02;
 1.20e-01  7.87e-01 -1.33e-01;
 5.84e-02 -1.33e-01  1.66e+00;]
solve(n, X, P8, Q8, M8, false)
@show SOS_time Newton_time