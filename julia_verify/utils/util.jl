using Symbolics
using DynamicPolynomials
using SumOfSquares
using MosekTools
using Printf
using LinearAlgebra
using JSON

function symplify_rational(num; prec=2)
    float_num = round(num, digits=prec)
    rationalize(BigInt, float_num)
end

magicalNumber = Rational{BigInt}(114514//19192024)
@Symbolics.variables magicalVar

function coefficients(f, vars)
    # 加上一个magicalNumber 保证一定存在常数项, 后续再减去, 这个操作为了保证表达式一定是多项式.
    # println("start coefficients function")
    f += magicalNumber
    D = Dict()
    constant = getfield(f.val, Symbol("###Any###3"))
    if constant != 0
        D[1] = constant
    end
    for (k, v) in getfield(f.val, Symbol("###Any###4"))
        t = 1
        for var in vars
            if Symbolics.isequal(var, 1)  # 防止死循环
                continue
            end
            while !('/' in string(k / var))
                t *= var
                k /= var
            end
        end
        haskey(D, t) ? D[t] += k * v : D[t] = k * v
    end
    D[1] -= magicalNumber
    if Symbolics.isequal(D[1], 0) # 删除系数为0的常数项
        delete!(D, 1)
    end
    D
end

function expNorm2(expr; p=2)
    ans = 0
    # 先给表达式加上一个"margicalVar"可以保证表达式一定为多项式.
    expr += magicalVar 
    # @show expr getfield(expr.val, Symbol("###Any###4"))
    for (k, v) in getfield(expr.val, Symbol("###Any###4"))
        if string(k) != "magicalVar"
        ans += v * v
        end
    end
    ans
end

function deleteColumn(matrix, j)
    n, m = size(matrix)
    if j > m
        throw(ArgumentError("Invalid column $j"))
    end
    matrix = matrix[:, 1:m .≠ j]
end

function convert_string2symbolics(str_expr)
    # 默认已经设置好了所有表达式所需的变量
    # 元编程很危险, 但暂时没找到更好的转变方式
    eval(Meta.parse(str_expr))
end

function solve_SOS(f)
    f = convert_string2symbolics(f)
    model = SOSModel(Mosek.Optimizer)
    con = @constraint(model, f >= 0)
    optimize!(model)
    @show termination_status(model)
    con
end


function read_example(file_name)
    list = []
    open(file_name, "r") do file
        for line in eachline(file)
            push!(list, line)
        end
    end
    return parse(Int, list[1]), list[2]
end

function save(file_name, data)
    open(file_name, "w") do io
        JSON.print(io, data, 2)
    end
end


