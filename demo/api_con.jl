using JuMP
using DynamicPolynomials
using SumOfSquares
using MosekTools # 或其他 SDP 求解器

# 1. 定义模型
model = Model(Mosek.Optimizer)

# 2. 定义多项式变量和参数
@polyvar x[1:3]  # 定义多项式变量 x₁, x₂, x₃
@variable(model, c[1:2])  # 定义参数 c₁, c₂

# 3. 初始化一个矩阵来存储约束
# 假设我们有两组约束，每组存储 2 个约束
rows = 2  # 组数
cols = 2  # 每组的约束数
constraints_matrix = Matrix{Any}(undef, rows, cols)  # 未初始化的矩阵

# 4. 定义多项式并添加约束到矩阵
# 第一组约束
p1 = x[1]^2 + c[1] * x[2] + 1
p2 = x[2]^2 + c[2] * x[1] + x[1] * x[3]

# 修复强凸性
@variable(model, ε >= 1e-6)  # 小正数
p1_corrected = p1 + ε
p2_corrected = p2 + ε

# 添加到第一组（第一行）
constraints_matrix[1, 1] = @constraint(model, p1_corrected in SOSCone())
constraints_matrix[1, 2] = @constraint(model, p2_corrected in SOSCone())

# 第二组约束
p3 = x[1]^4 + x[2]^4 + c[1] * c[2]
p4 = x[1]^2 * x[2]^2 + c[2] * x[3]

# 添加到第二组（第二行）
constraints_matrix[2, 1] = @constraint(model, p3 in SOSCone())
constraints_matrix[2, 2] = @constraint(model, p4 in SOSCone())

# 5. 求解模型
optimize!(model)

# 6. 输出矩阵中的约束
if termination_status(model) == MOI.OPTIMAL
    println("求解成功！")
    println("所有约束存储在矩阵中：")
    for i in 1:rows
        for j in 1:cols
            println("约束 ($i, $j): ", constraints_matrix[i, j])
        end
    end
else
    println("未找到满足条件的解。")
end
