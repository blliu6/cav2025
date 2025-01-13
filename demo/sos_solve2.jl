using JuMP
using DynamicPolynomials
using SumOfSquares
using MosekTools # 或其他 SDP 求解器

# 1. 定义多项式变量
@polyvar x[1:2]  # 定义两个多项式变量 x₁, x₂

# 2. 创建模型
model = Model(Mosek.Optimizer)

# 3. 定义一个多项式 p(x)
p = x[1]^4 + 2 * x[1]^2 * x[2]^2 + x[2]^4 - x[1]^2 * x[2] + 0.5

# 4. 添加 SOS 约束
con = @constraint(model, p in SOSCone())

# 5. 求解模型
optimize!(model)

# 6. 提取 SOS 分解
if termination_status(model) == MOI.OPTIMAL
    println("求解成功，提取 SOS 分解：")

    # 提取分解的单项式列表 z(x) 和矩阵 Q
    decomposition = gram_matrix(p, model)
    monomials = decomposition.monomials  # 单项式向量 z(x)
    Q = decomposition.Q  # 分解矩阵 Q

    # 输出单项式列表
    println("单项式列表 z(x):")
    println(monomials)

    # 输出 SOS 分解矩阵 Q
    println("SOS 分解矩阵 Q:")
    println(Q)
else
    println("未找到满足条件的解，无法提取 SOS 分解。")
end
