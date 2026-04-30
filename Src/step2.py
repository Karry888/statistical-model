import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("最终建模面板数据.csv", index_col=["province_code", "year"])

# ---------------------- 关键修复：只保留数值型变量 ----------------------
numeric_cols = [
    "output", "export", "gd_al_product_output", "policy_shock",
    "elec_price", "grid_cef", "policy_dummy", "al_price",
    "treat", "post", "did"
]
df_num = df[numeric_cols]

# ---------------------- 1. 描述性统计 ----------------------
print("="*50)
print("📊 数据描述性统计")
print("="*50)
print(df_num.describe().round(2))
df_num.describe().to_excel("描述性统计.xlsx")

# ---------------------- 2. 相关性热力图 ----------------------
plt.figure(figsize=(10,6))
sns.heatmap(df_num.corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("变量相关性热力图")
plt.tight_layout()
plt.savefig("相关性图.png", dpi=300)
plt.show()

# ---------------------- 3. 产量趋势图 ----------------------
df_reset = df.reset_index()
df_reset["group"] = np.where(df_reset["treat"]==1, "处理组", "对照组")
trend = df_reset.groupby(["year","group"])["output"].mean().unstack()

plt.figure(figsize=(12,5))
trend.plot(marker="o", ax=plt.gca())
plt.axvline(2020, color="red", linestyle="--", label="2020政策年")
plt.title("处理组 vs 对照组 产量趋势")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("产量趋势图.png", dpi=300)
plt.show()

print("\n✅ 第二步完成！所有文件已保存！")