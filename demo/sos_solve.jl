include("utils/util.jl")
include("utils/newtonUpdate.jl")
include("utils/rationalSOS.jl")


X = nothing
eig_tol = 1e-4
tol = 1e-50

# 求解添加乘子后的问题
function solve_SOS2(x, f, deg=4)
    # @show n f
    f = convert_string2symbolics(f) # 转成符号表达式
    model = SOSModel(Mosek.Optimizer)
    poly, par, terms = polynomial(model, x, deg) # 多项式模板
    # @show poly par terms
    # @show f*poly
    con_all = @constraint(model, f * poly >= 0) # sos constraint
    con_poly = @constraint(model, poly >= 0)
    optimize!(model)
    # @show con_all con_poly 
    @show termination_status(model)

    con_all, con_poly
end

# 定义符号变量
function define_symbols(n)
    @polyvar x[1:n]
    return x
end

# 生成多项式
function polynomial(model, x, deg=2)
    # TODO: need to fix, stay here, becouse of tight time.
    if deg == 2 && length(x) > 8
        parameters = []
        terms = []
        poly = 0
        var_count = 1  # 用于计数变量

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

        return poly, parameters, terms
    else
        parameters = []
        terms = []
        
        exponents = collect(Iterators.product([(0:deg) for _ in 1:length(x)]...))  # 生成所有可能的指数组合
        exponents = filter(e -> sum(e) <= deg, exponents)  # 移除指数和大于 deg 的组合
        poly = 0
        par = @variable(model, par[1:length(exponents)])

        for (i, e) in enumerate(exponents)
            term = prod([x[j]^exp for (j, exp) in enumerate(e)])
            push!(terms, term)
            # @show term par[i]
            poly += par[i] * term
        end

        return poly, par, terms
    end
end

function solve(file_name, newton)
    global X

    n, strExpr = read_example(file_name)

    @polyvar X[1:n]
    con, con_poly = solve_SOS2(X, strExpr, 0)
    gram = gram_matrix(con)
    @show gram.Q
    Q = round.(gram.Q, digits=2)
    rQ = symplify_rational.(Q)
    
    mono = gram.basis.monomials
    
    strMono = string.(mono)
    @show strMono
    @Symbolics.variables X[1:n]
    f = convert_string2symbolics(strExpr)
    mono = convert_string2symbolics.(strMono)
   

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
        "expression" => strExpr,
        "coeff" => join([string(c) for c in coeffs], ","),
        "term" => join([string(t) for t in terms], ","),
        "SOSrational_time" => SOSrational_time,
        "SOS_expression" => string(coeffs' * terms.^2)
    )

    # save("src/output/data.json", data)
    
end
    # -

