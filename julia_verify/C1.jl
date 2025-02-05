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
## Total Decomposition
## Multiplier-1
P0 = 5349*X[1]^2//500 + 2387*X[1]*X[2]//2000 + 8539*X[1]//1000 + 92937*X[2]^2//10000 + 27453*X[2]//5000 + 5047//500
M0 = [1, X[2], X[1]]
Q0 = [ 1.01e+01  2.74e+00  4.27e+00;
 2.74e+00  9.29e+00  5.93e-01;
 4.27e+00  5.93e-01  1.07e+01;]

rational_SOS(P0, 1, 0, M0, Q0)

solve(n, X, P0, Q0, M0, false)
P1 = 10699*X[1]^4//1000 + 11707*X[1]^3*X[2]//10000 - 2857*X[1]^3//200 + 9989*X[1]^2*X[2]^2//500 - 62953*X[1]^2*X[2]//10000 + 6091*X[1]^2//500 + 2947*X[1]*X[2]^3//2500 - 95641*X[1]*X[2]^2//10000 - 3463*X[1]*X[2]//500 - 14539*X[1]//1000 + 46479*X[2]^4//5000 - 56683*X[2]^3//10000 + 41351*X[2]^2//1000 - 65487*X[2]//10000 + 3101//250
M1 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q1 = [ 1.24e+01 -3.27e+00 -7.27e+00  3.28e+00 -1.92e+00 -2.06e+00;
-3.27e+00  3.48e+01 -1.54e+00 -2.82e+00 -1.61e+00 -1.36e+00;
-7.27e+00 -1.54e+00  1.63e+01 -3.19e+00 -1.79e+00 -7.14e+00;
 3.28e+00 -2.82e+00 -3.19e+00  9.29e+00  5.93e-01  1.00e+00;
-1.92e+00 -1.61e+00 -1.79e+00  5.93e-01  1.80e+01  5.93e-01;
-2.06e+00 -1.36e+00 -7.14e+00  1.00e+00  5.93e-01  1.07e+01;]

solve(n, X, P1, Q1, M1, false)
# unsafe
## Total Decomposition
P2 = 30013*X[1]^4//1000 + 27673*X[1]^3*X[2]//10000 - 2519*X[1]^3//100 + 54637*X[1]^2*X[2]^2//1000 - 1577*X[1]^2*X[2]//125 + 16131*X[1]^2//5000 + 13979*X[1]*X[2]^3//5000 - 97*X[1]*X[2]^2//5 - 5243*X[1]*X[2]//200 - 6281*X[1]//200 + 6147*X[2]^4//250 - 25989*X[2]^3//10000 + 231*X[2]^2//8 - 1509*X[2]//625 + 7549//250
M2 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q2 = [ 3.02e+01 -1.21e+00 -1.57e+01 -2.01e+00 -9.33e+00 -1.40e+01;
-1.21e+00  3.29e+01 -3.76e+00 -1.30e+00 -5.09e+00 -3.67e+00;
-1.57e+01 -3.76e+00  3.12e+01 -4.62e+00 -2.63e+00 -1.26e+01;
-2.01e+00 -1.30e+00 -4.62e+00  2.46e+01  1.39e+00  5.24e+00;
-9.33e+00 -5.09e+00 -2.63e+00  1.39e+00  4.41e+01  1.39e+00;
-1.40e+01 -3.67e+00 -1.26e+01  5.24e+00  1.39e+00  3.00e+01;]
solve(n, X, P2, Q2, M2, false)
## Multiplier-1
P3 = 30013*X[1]^2//1000 + 27897*X[1]*X[2]//10000 + 5711*X[1]//200 + 12293*X[2]^2//500 + 20299*X[2]//5000 + 33833//1000
M3 = [1, X[2], X[1]]
Q3 = [ 3.38e+01  2.02e+00  1.43e+01;
 2.02e+00  2.46e+01  1.39e+00;
 1.43e+01  1.39e+00  3.00e+01;]
solve(n, X, P3, Q3, M3, false)
# Lie
## Multiplier-2
P4 = 212*X[1]^2//625 + 3463*X[1]*X[2]//5000 + 251*X[1]//500 + 291*X[2]^2//200 + 101*X[2]//250 + 241//500
M4 = [1, X[2], X[1]]
Q4 = [ 4.82e-01  2.02e-01  2.51e-01;
 2.02e-01  1.45e+00  3.46e-01;
 2.51e-01  3.46e-01  3.39e-01;]
solve(n, X, P4, Q4, M4, false)
## Total Decomposition
P5 = 871*X[1]^4//2000 + 1643*X[1]^3*X[2]//50000 + 8563*X[1]^3//10000 + 3433*X[1]^2*X[2]^2//5000 + 1253*X[1]^2*X[2]//1000 + 5903*X[1]^2//10000 + 1361*X[1]*X[2]^3//5000 - 1191*X[1]*X[2]^2//500 + 897*X[1]*X[2]//500 + 507*X[1]//2000 + 143*X[2]^4//100 + 1099*X[2]^3//10000 + 619*X[2]^2//250 - 5373*X[2]//10000 + 1163//1000
M5 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q5 = [ 1.16e+00 -2.68e-01  1.27e-01  1.27e-01  4.61e-01 -2.47e-01;
-2.68e-01  2.22e+00  4.34e-01  5.56e-02 -6.63e-01  3.21e-01;
 1.27e-01  4.34e-01  1.09e+00 -5.27e-01  3.06e-01  4.29e-01;
 1.27e-01  5.56e-02 -5.27e-01  1.43e+00  1.37e-01 -2.90e-01;
 4.61e-01 -6.63e-01  3.06e-01  1.37e-01  1.27e+00  1.69e-02;
-2.47e-01  3.21e-01  4.29e-01 -2.90e-01  1.69e-02  4.36e-01;]
solve(n, X, P5, Q5, M5, false)
## Multiplier-1
P6 = 4821*X[1]^2//5000 + 379*X[1]*X[2]//200 + 49*X[1]//200 + 59*X[2]^2//25 - 161*X[2]//5000 + 537//1000
M6 = [1, X[2], X[1]]
Q6 = [ 5.37e-01 -1.63e-02  1.22e-01;
-1.63e-02  2.36e+00  9.46e-01;
 1.22e-01  9.46e-01  9.63e-01;]
solve(n, X, P6, Q6, M6, false)

@show SOS_time Newton_time