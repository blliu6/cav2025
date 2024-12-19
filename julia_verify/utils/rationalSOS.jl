include("util.jl")

function rational_SOS(f, g, minval, monos, Q)
    start_time = time()
    # Convert input variables to symbolic expressions
    @Symbolics.variables _r _p[1:length(monos)] _q[1:length(monos)]

    rf, rg, rm, rQ = f, g, minval, Q

    # Create a vector of monomials
    mv = Vector(monos)

    # Check the dimensions of Q and mv
    m, t = size(Q)
    if m != t || m != length(mv)
        error("Invalid input.")
    end

    # Compute the difference
    t = Symbolics.expand(rf - rm * rg - mv' * rQ * mv)
    # println("the distance is \n$t\n")

    # Add the product of _p and _q
    t = t * _r + sum(_p[i] * mv[i] for i in 1:m) * sum(_q[i] * mv[i] for i in 1:m)
    # println("after add the product of _q and _p, t is \n$t\n")

    # Expand and Extract coefficients
    D = coefficients(Symbolics.expand(t), mv)
    coeffs = collect(values(D))
    # println(coeffs)
    # Extract the coefficients of _r
    b = [Symbolics.coeff(i, _r) for i in coeffs]
    # println("after extract the coefficients of _r is \n$b \n")

    # Subtract b*_r from each element of coeffs
    coeffs = coeffs .- b .* _r
    # println("Subtract b*_r from each element of coeffs: \n$coeffs\n")

    # Normalize and sum the coefficients
    coeffs = Symbolics.expand(sum(b[i] * coeffs[i] / expNorm2(coeffs[i]) for i in 1:lastindex(coeffs)))
    # println("Normalize and sum the coefficients: \n$coeffs\n")

    # Extract the coefficients of _p
    coeffs_p = [Symbolics.coeff(coeffs, _p[i]) for i in 1:m]
    # println("coeff whith p:\n $(coeffs_p)\n")

    # @show coeffs_p
    # Generate the matrix equation
    A, b = generate_matrix(coeffs_p, [_q[i] for i in 1:m])

    # Update rQ
    rQ += A

    # @show rQ
    # Print the distance between Q and rQ
    println("\nThe distance between Q and rQ is $(LinearAlgebra.norm(Float64.(A)))\n")

    # Symmetrize rQ
    rQ = (transpose(rQ) + rQ) / 2
    # println("Symmetrize rQ is \n$(rQ)\n")
    # Process the matrix rQ
    d = zeros(Rational{BigInt}, m)
    for i in 1:m
        t = rQ[i, i]
        if t > 0
            d[i] = 1 / t
            for j in i+1:m
                for k in j:m
                    rQ[j, k] -= rQ[i, j] * rQ[i, k] / t
                end
            end
            
        elseif norm(rQ[i, i:end]) > 0
            println("rQ is not positive semidefinite.")
            println(norm(rQ[i, i:end]))
            return rm, d, rQ * mv
        end
        rQ[i+1:end, i] .= 0
    end

    # Print results
    totalTime = time() - start_time
    println("$rm is the lower bound.")
    println("$totalTime seconds")
    
    return rm, totalTime, d, rQ * mv
end

# Utility function to generate matrix and vector
function generate_matrix(coeffs, vars)
    A = zeros(Rational{BigInt}, length(coeffs), length(vars))
    b = []

    for i in 1:lastindex(coeffs)
        for j in 1:lastindex(vars)
            bf = Symbolics.coeff(coeffs[i], vars[j])
            # @show bf vars[j]
            try
                A[i, j] = convert(Rational{BigInt}, bf)
            catch
                A[i, j] = convert(Rational{BigInt}, getfield(bf, Symbol("###Any###3")))
                @show bf getfield(bf, Symbol("###Any###4"))
                @show getfield(bf, Symbol("###Any###3"))
                @show A[i, j]
            end
        end
        push!(b, coeffs[i])
    end
    return A, b
end
# -