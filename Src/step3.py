import pandas as pd
from linearmodels import PanelOLS

# 读取数据
df = pd.read_csv("最终建模面板数据.csv", index_col=["province_code", "year"])

# 只保留建模用的数值变量
model_vars = [
    "output", "export", "gd_al_product_output", "policy_shock",
    "elec_price", "grid_cef", "policy_dummy", "al_price"
]
df_model = df[model_vars].dropna()

# ==================== 固定效应回归模型 ====================
# 公式：被解释变量 ~ 解释变量 + 控制变量 + 个体固定效应
formula = """
output ~ 1 + export + gd_al_product_output + policy_shock 
+ elec_price + grid_cef + policy_dummy + al_price 
+ EntityEffects
"""

# 运行回归
model = PanelOLS.from_formula(formula, data=df_model)
result = model.fit(cov_type="clustered", cluster_entity=True)

# 打印结果
print("="*60)
print("🏆 基准回归结果（个体固定效应）")
print("="*60)
print(result)

# 保存结果到文本（直接复制到论文）
with open("基准回归结果.txt", "w", encoding="utf-8") as f:
    f.write(result.summary.as_text())

print("\n✅ 第三步完成！回归结果已保存为：基准回归结果.txt")