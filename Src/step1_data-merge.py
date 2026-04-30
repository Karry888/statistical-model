import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 全局绘图设置（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ==================== 1. 读取3个清洗后的CSV文件 ====================
# 【重要】请把3个CSV文件和这个py代码文件放在同一个文件夹里！
df_y = pd.read_csv(r"D:\统计建模\数据\data\statistical-model\analysis\Y_Output_cleaned.csv")  # 被解释变量：产量
df_c = pd.read_csv(r"D:\统计建模\数据\data\statistical-model\analysis\C_Control_Variable_cleaned.csv")  # 控制变量
df_x = pd.read_csv(r"D:\统计建模\数据\data\statistical-model\analysis\X_Guangdong_Demand_and_Policy_cleaned.csv")  # 广东核心解释变量

# ==================== 2. 数据清洗与规范 ====================
# 2.1 被解释变量Y：过滤有效样本，规范数据类型，剔除单位说明行
df_y = df_y[df_y["year"].between(2015, 2025)].copy()
df_y["year"] = df_y["year"].astype(int)
# 仅保留建模需要的核心列
df_y = df_y[["province_code", "province_name", "year", "output"]]

# 2.2 控制变量C：规范数据类型，剔除冗余列
df_c = df_c[df_c["year"].between(2015, 2025)].copy()
df_c["year"] = df_c["year"].astype(int)
# 仅保留建模需要的核心控制变量
df_c = df_c[["province_code", "year", "elec_price", "grid_cef", "policy_dummy", "al_price"]]

# 2.3 核心解释变量X：规范数据类型，剔除冗余列
df_x = df_x[df_x["year"].between(2015, 2025)].copy()
df_x["year"] = df_x["year"].astype(int)
# 仅保留建模需要的核心解释变量
df_x = df_x[["year", "gd_al_product_output", "export", "policy_shock"]]

# ==================== 3. 合并为完整面板数据集 ====================
# 第一步：合并Y（产量）和C（控制变量），主键：province_code + year
df_merge1 = pd.merge(
    df_y,
    df_c,
    on=["province_code", "year"],
    how="left",
    validate="1:1"
)

# 第二步：合并X（广东年度变量），主键：year（同一年份所有省份共享广东数据）
df_full = pd.merge(
    df_merge1,
    df_x,
    on="year",
    how="left"
)

# ==================== 4. 面板数据标准化设置 ====================
# 设置面板数据索引：个体维度(province_code) + 时间维度(year)
df_full = df_full.set_index(["province_code", "year"]).sort_index()

# 生成DID模型所需变量
# treat：处理组虚拟变量（YN/GX/NMG=1，对照组HN/SD=0）
df_full["treat"] = np.where(df_full.index.get_level_values("province_code").isin(["YN", "GX", "NMG"]), 1, 0)
# post：政策时点虚拟变量（2020年及以后=1，之前=0）
df_full["post"] = np.where(df_full.index.get_level_values("year") >= 2020, 1, 0)
# did交互项：核心DID变量
df_full["did"] = df_full["treat"] * df_full["post"]

# ==================== 5. 数据完整性校验 ====================
print("="*60)
print("✅ 最终面板数据校验结果")
print("="*60)
print(f"样本总量：{len(df_full)}行")
print(f"覆盖省份：{df_full.index.levels[0].tolist()}")
print(f"时间跨度：{df_full.index.levels[1].min()} - {df_full.index.levels[1].max()}年")
print(f"平衡面板校验：{len(df_full) == 5 * 11}")  # 5省*11年=55个样本
print("\n📊 全变量缺失值统计：")
print(df_full.isnull().sum())
print("\n📈 数据前5行预览：")
print(df_full.head())
print("="*60)

# 保存最终建模用面板数据，方便后续调用
df_full.to_csv("最终建模面板数据.csv", encoding="utf-8-sig")
print("✅ 最终面板数据已保存为：最终建模面板数据.csv")