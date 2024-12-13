include("util.jl")
using Symbolics
using Printf

function newton_refine_update(f, monos, Q, eig_tol, max_iter, tol, indets)
    # Convert the list of monomials into a vector
    mv = Vector(monos)

    # Get the dimensions of the matrix Q
    m, t = size(Q)

    # Check if the dimensions of Q and mv match
    if m ≠ t || m ≠ length(mv)
        throw(ArgumentError("Invalid input."))
    end

    # Print the norm of the difference between f and m' * Q * m
    println("|f - m'*Q*m|: ", sqrt(expNorm2(expand(f - mv' * Q * mv))))

    # Print the monomials vector
    # println("Monomials: ", mv)

    # Record the current time for timing purposes
    tm = time()

    # Compute the eigenvalues and eigenvectors of Q
    t, L = eigen(Q)

    # Initialize counters
    r = 0
    k::Int64 = 0

    # Initialize the vector v and the matrix sL
    v = zeros(BigFloat, m * m)
    @Symbolics.variables _q[1:m * m]
    sL = zeros(Num, m, m)
    # Process the eigenvalues and eigenvectors
    for j in 1:lastindex(t)
        L[:, j] .= L[:, j] * sqrt(abs(t[j]))  # Scale the eigenvectors
        if abs(t[j]) > eig_tol
            r += 1
            for i in 1:m
                if abs(L[i, j]) > 1e-3
                    k += 1
                    v[k] = real(L[i, j])
                    sL[i, j] = _q[k]  # Placeholder for the symbol _q
                end
            end
        end
    end
    # @show sL
    
    # Print the rank deficiency of Q
    println("the rank deficiency of Q: $r")

    # Trim v to the used portion
    v = v[1:k]

    # Compute the vector t as the product of the monomials vector and sL
    t = mv' * sL

    # Compute the vector F as the coefficients of the expanded difference
    D = coefficients(expand(f - sum(i^2 for i in t)), indets)
    F = collect(Num, values(D))
    
    # Compute the Jacobian matrix W of F with respect to the symbols _q
    W = Symbolics.jacobian(F, [_q[i] for i in 1:k])
    
    # Evaluate F at the initial point v
    b = Symbolics.substitute(F, Dict([_q[i]=>v[i] for i in 1:lastindex(v)]))

    # Compute the norm of b
    t = norm(b, 2)

    # Print the original residue without Newton iteration
    println("the original residue without Newton iteration: $t")

    # Print the dimension of the Jacobian matrix
    println("the dimension of Jacobian matrix: $(size(W))")

    old = Base.precision(BigFloat)
    # Iterate using Newton's method
    
    iter_cnt = 0

    for i in 1:max_iter
        if t ≤ tol
            break
        end
        iter_cnt = i
        Base.setprecision(BigFloat, old*4)
        A = Symbolics.substitute(W, Dict([_q[i]=>v[i] for i in 1:lastindex(v)]))
        
        try
            v .-=  A\b # Update v using the least squares solution
        catch e
            if isa(e, SingularException)
                v .-= pinv(A) * b
            end
        end
        # Evaluate F at the updated point v
        b = Symbolics.substitute(F, Dict([_q[i]=>v[i] for i in 1:lastindex(v)]))
        
        Base.setprecision(BigFloat, old)
        # Compute the norm of b
        t = norm(b, 2)

        # Print the updated residue
        @printf("iter%d: |f - ∑ f_i^2|: %.50f\n", iter_cnt, t)

        # Break the loop if the tolerance is met
        if t ≤ tol
            break
        end
    end
    
    # Print the final residue
    # @printf("iter%d: |f - ∑ f_i^2|: %.50f\n",iter_cnt, t)
    
    # Evaluate sL at the final point v
    sL = Symbolics.substitute(sL, Dict([_q[i]=>v[i] for i in 1:lastindex(v)]))

    # Update sL to remove columns with zero norm
    sL_update = sL
    
    for i in reverse(1:size(sL, 2))
        if norm(sL[:, i], 2) ≈ 0
            sL_update = deleteColumn(sL_update, i)
        end
    end

    return sL_update
end
