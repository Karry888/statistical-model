import pandas as pd
from linearmodels import PanelOLS
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 强制设置中文字体（解决乱码核心） ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先用黑体/微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'  # 强制使用无衬线字体

# 读取数据
df = pd.read_csv("最终建模面板数据.csv", index_col=["province_code", "year"])

# ==================== 标准DID模型 ====================
formula_did = """
output ~ did + elec_price + grid_cef + policy_dummy + al_price
+ EntityEffects
"""

model_did = PanelOLS.from_formula(formula_did, data=df, drop_absorbed=True)
result_did = model_did.fit(cov_type="clustered", cluster_entity=True)

# 输出结果
print("="*60)
print("🎯 DID双重差分模型结果（最终版）")
print("="*60)
print(result_did)

# 保存结果
with open("DID回归结果.txt", "w", encoding="utf-8") as f:
    f.write(result_did.summary.as_text())

# ==================== 修复版绘图（中文正常显示） ====================
df_reset = df.reset_index()
df_reset["group"] = np.where(df_reset["treat"] == 1, "处理组", "对照组")
trend = df_reset.groupby(["year", "group"])["output"].mean().unstack()

plt.figure(figsize=(12,5))
trend.plot(marker="o", linewidth=2, ax=plt.gca())
plt.axvline(2020, color="red", linestyle="--", label="2020政策冲击")
plt.title("DID 处理组 vs 对照组 产量趋势", fontsize=14)
plt.ylabel("电解铝产量（万吨）", fontsize=12)
plt.xlabel("年份", fontsize=12)
plt.legend(["对照组", "处理组", "2020政策冲击"])  # 手动指定图例，避免乱码
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("DID趋势图.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n✅ 第四步成功运行！文件已保存！")