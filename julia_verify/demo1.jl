include("utils/util.jl")
include("utils/newtonUpdate.jl")
include("utils/rationalSOS.jl")


X = nothing
eig_tol = 1e-4
tol = 1e-50


n = 2
@polyvar X[1:n]

# 求解添加乘子后的问题
function solve_SOS2(x, f, deg=4)
    # @show n f
    f = convert_string2symbolics(f)
    model = SOSModel(Mosek.Optimizer)
    poly, _, _ = polynomial(model, x, deg)
    con_all = @constraint(model, f * poly >= 0)
    con_poly = @constraint(model, poly >= 0)
    optimize!(model)
    
    @show termination_status(model)

    con_all, con_poly
end

function solve_positive_with_box(f, x, con, deg=2)
    @show f x
    expr = convert_string2symbolics(f)
    model = SOSModel(Mosek.Optimizer)
    con_polys = []
    for (i, c) in enumerate(con)
        P, _, _ = polynomial(model, x, deg)
        @show P
        con_poly = @constraint(model, P >= 0)
        push!(con_polys, con_poly)
        @show c[1] c[2]
        expr = expr - P * (c[1] - x[i]) * (c[2] - x[i])
        @show expr
    end
    @show "YE"
    # con_expr = @constraint(model, expr >= 0)
    
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

# 定义符号变量
function define_symbols(n)
    @polyvar x[1:n]
    return x
end

# 生成多项式
var_count = 1

function polynomial(model, x, deg=2)
    global var_count
    @show "begin polynomial" var_count
    # TODO: need to fix, stay here, becouse of tight time.
    if deg == 2 && length(x) > 8
        parameters = []
        terms = []
        poly = 0
        # var_count = 1  # 用于计数变量

        # 添加常数项
        push!(parameters, @polyvar parameter[var_count])
        var_count += 1
        push!(terms, 1)
        poly += parameters[end]

        # 添加线性项
        for i in eachindex(x)
            push!(parameters, @polyvar parameter[var_count])
            var_count += 1
            push!(terms, x[i])
            poly += parameters[end] * terms[end]

            push!(parameters, @polyvar parameter[var_count])
            var_count += 1
            push!(terms, x[i]^2)
            poly += parameters[end] * terms[end]
        end

        # 添加二次交叉项
        for i in eachindex(x)
            for j in i+1:length(x)
                push!(parameters, @polyvar parameter[var_count])
                var_count += 1
                push!(terms, x[i] * x[j])
                poly += parameters[end] * terms[end]
            end
        end
        @show var_count
        return poly, parameters, terms
    else
        parameters = []
        terms = []
        
        exponents = collect(Iterators.product([(0:deg) for _ in 1:length(x)]...))  # 生成所有可能的指数组合
        exponents = filter(e -> sum(e) <= deg, exponents)  # 移除指数和大于 deg 的组合
        poly = 0
        pars = @variable(model, pars[var_count:var_count+length(exponents)])

        for (i, e) in enumerate(exponents)
            term = prod([x[j]^exp for (j, exp) in enumerate(e)])
            push!(terms, term)
            # @show term par[i]
            poly += pars[i] * term
        end
        
        @show var_count
        return poly, pars, terms
    end
end

# 1. 对于一个区域所需要的一个BC， 区间， 好几个乘子


function solve(n, strExpr, newton)
    global X

    con, con_poly = solve_SOS2(X, strExpr, 2)
    gram = gram_matrix(con)
    
    Q = round.(gram.Q, digits=2)
    rQ = symplify_rational.(Q)
        
    mono = gram.basis.monomials
    strMono = string.(mono)
    
    @Symbolics.variables X[1:n]
    f = convert_string2symbolics(strExpr)
    mono = convert_string2symbolics.(strMono)

    model = nothing
    if !newton
        model = rational_SOS(f, 1, 0, mono, rQ)
    else
        update_Q = newton_refine_update(f, mono, Q, eig_tol, 200, tol, X)
        Q_res = symplify_rational.(update_Q*update_Q')
        model = rational_SOS(f, 1, 0, mono, Q_res)
    end
    
    SOSrational_time, coeffs, terms = model[2:end]
    
    @show expand(f-(coeffs' * terms.^2))


    data = Dict(
        "n" => n,
        "expression" => strExpr,
        "coeff" => join([string(c) for c in coeffs], ","),
        "term" => join([string(t) for t in terms], ","),
        "SOSrational_time" => SOSrational_time,
        "SOS_expression" => string(coeffs' * terms.^2)
    )

    # save("src/output/data.json", data)
    
end

#B =  "3.28824684375188*X[1] + 12.9600997744408*X[2] + 3.51405697166029"


# 2. 定义多项式 f
f = "X[1]^2"  # 要验证的多项式 f

# 3. 定义约束 con (例如，x 和 y 在 [-1, 1] 之间)
con = [[-1, 1], [-1, 1]]  # 变量 x 和 y 都在区间 [-1, 1] 中

# 4. 调用 `solve_positive_with_box` 函数来验证 f 是否在给定约束下是正定的
result = solve_positive_with_box(f, X, con)

# 5. 打印结果
println("Is the polynomial positive under the constraints? ", result)