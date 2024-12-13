using JuMP
using SumOfSquares
using Mosek
using DynamicPolynomials
using Symbolics
include("utils/util.jl")

"""
    verify_positive(expr, con; deg=2)

Verifies whether the polynomial `expr` is positive under the constraints `con` using
Sum of Squares (SOS) programming. The degree of the SOS polynomials can be specified
with the `deg` parameter (default is 2).

# Arguments
- `expr`: The polynomial expression to verify for positivity.
- `con`: A vector of polynomial constraints.
- `deg` (optional): The degree of the SOS polynomials (default is 2).

# Returns
- `Bool`: `true` if the SOS program is feasible (i.e., `expr` is positive under `con`), 
         `false` otherwise.
"""
function verify_positive(expr, con; deg=2)
    # Extract variables from the expression and constraints
    vars = Symbolics.variables(expr)
    for c in con
        vars = union(vars, Symbolics.variables(c))
    end

    # Convert Symbolic variables to MultivariatePolynomials variables
    poly_vars = [symbol_to_polynomial_var(v) for v in vars]

    # Create a JuMP model with MOSEK as the optimizer
    model = Model(Mosek.Optimizer)


    # Initialize the current expression
    current_expr = expr

    # Iterate over each constraint and add corresponding SOS constraints
    for c in con
        # Generate all monomials up to degree `deg` for the variables
        monomials_c = monomials(poly_vars, deg)

        # Define SOS polynomial P for the current constraint
        # P is a generic SOS polynomial of degree `deg`
        P, par, terms = polynomial(model, poly_vars, deg)
        # Add SOS constraint: P is a sum of squares
        @constraint(model, P >= 0)

        # Update the expression: expr = expr - c * P
        current_expr = current_expr - c * P
    end

    # Expand the final expression
    current_expr = Symbolics.expand(current_expr)

    # Add SOS constraint on the final expression
    @constraint(model, current_expr >= 0)

    # Optimize the model
    try
        optimize!(model)
        status = termination_status(model)
        if status == MOSEK.OPTIMAL
            return true
        else
            return false
        end
    catch e
        # In case of any errors (e.g., infeasibility), return false
        return false
    end
end

"""
    symbol_to_polynomial_var(sym::Symbol)

Converts a Symbolics.Symbol to a MultivariatePolynomials Variable.

# Arguments
- `sym`: The symbolic variable to convert.

# Returns
- `Variable`: A MultivariatePolynomials variable corresponding to `sym`.
"""
function symbol_to_polynomial_var(sym::Symbol)
    return PolynomialVar(sym)
end


using Symbolics

# Define symbolic variables
@variables x y

# Define the expression you want to verify
expr = x^4 + y^4 + 1

# Define constraints (e.g., x^2 + y^2 - 1 >= 0)
con = [x^2 + y^2 - 1]

# Verify positivity
is_positive = verify_positive(expr, con, deg=2)

println("Is the expression positive under the constraints? ", is_positive)
