"""
数据清洗脚本 (Data_Cleaning.py)
功能：读取Data目录下的Excel数据，清洗后保存为CSV格式。
主要清洗步骤：
1. 读取指定的Excel文件及其第一个工作表。
2. 移除包含“单位”或“数据单位”的行（这些是数据说明，非观测值）。
3. （可选）对数值型列的缺失值进行科学预测（线性插值/时间序列外推）。
4. 将清洗后的数据保存为CSV文件到新目录。
作者：元宝
日期：2026-04-30
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')  # 过滤简单警告，使输出更清晰

# ==================== 配置区域 ====================
# 1. 定义要处理的文件列表（基于您的文档）
DATA_FILES = [
    "C_Control_Variable.xlsx",
    "X_Guangdong_Demand_and_Policy.xlsx", 
    "Y_Output.xlsx"
]

# 2. 控制是否开启缺失值科学预测功能
ENABLE_MISSING_VALUE_PREDICTION = True  # 设置为 False 可关闭此功能

# 3. 定义目录路径 (假设脚本在 /Src/Data_Cleaning 下运行)
SCRIPT_DIR = Path(__file__).parent  # /Src/Data_Cleaning
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # 项目根目录 /
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = DATA_DIR / "Data_Cleaned"

# 4. 特定于文件的配置
# 对于每个文件，标识“单位行”可能存在的列名模式
UNIT_ROW_IDENTIFIERS = {
    "C_Control_Variable.xlsx": ["数据", "单位"],  # 最后一行“数据 单位”是单位行
    "X_Guangdong_Demand_and_Policy.xlsx": ["数据", "数据单位"], # 最后一行是单位行
    "Y_Output.xlsx": ["数据", "数据单位"] # 最后一行是单位行
}

# 定义每个文件中需要进行缺失值插值的时间序列列（如果开启预测）
# 结构: {文件名: {工作表名: [时间序列列名]}}
TIME_SERIES_COLS = {
    "Y_Output.xlsx": {
        "Sheet1": ["output"]  # 对产量进行插值和外推
    },
    "C_Control_Variable.xlsx": {
        "Sheet1": ["elec_price", "grid_cef"]  # 示例：对电价和碳因子进行插值
    }
    # 对于X文件，目前没有明确需要跨年插值的列 (export 2025年缺失，但属于面板外预测，暂不处理)
}
# ==================== 函数定义 ====================

def create_directories():
    """创建必要的输出目录"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[信息] 输出目录已就绪: {OUTPUT_DIR}")

def remove_unit_rows(df, file_name):
    """
    从DataFrame中移除包含“单位”信息的行。
    逻辑：查找第一列包含特定标识符（如“数据”、“单位”）的行并删除。
    """
    original_len = len(df)
    if file_name in UNIT_ROW_IDENTIFIERS:
        # 获取该文件对应的标识符列表
        identifiers = UNIT_ROW_IDENTIFIERS[file_name]
        # 构建一个条件：如果第一列的值是标识符列表中的任何一个，则标记为需要删除的行
        # 注意：DataFrame读取时，原表头的列名可能变成第一行数据。我们检查第一列（iloc[:, 0]）
        if not df.empty and df.columns[0] in identifiers:
            # 如果标识符出现在列名中，则可能是格式问题，尝试查找第一行
            mask = df.iloc[:, 0].astype(str).str.contains('|'.join(identifiers), na=False)
        else:
            # 通常，单位行是数据的一部分，检查第一列的值
            mask = df.iloc[:, 0].astype(str).str.contains('|'.join(identifiers), na=False)
        df_cleaned = df[~mask].copy()
        removed_count = original_len - len(df_cleaned)
        if removed_count > 0:
            print(f"  └─ 已移除 {removed_count} 行单位说明行。")
        return df_cleaned
    return df

def predict_missing_values(df, file_name, sheet_name, time_col='year', group_col='province_code'):
    """
    对面板数据中的缺失值进行科学预测（插值/有限外推）。
    策略：
    1. 按分组（如省份）对时间序列列进行插值。
    2. 使用线性插值填充中间缺失值。
    3. 对于首尾的缺失值，使用最近的有效值进行前向/后向填充（简单外推）。
    注意：这仅适用于具有明显时间趋势且缺失不多的序列。大量缺失或非时间序列列不适用。
    """
    if not ENABLE_MISSING_VALUE_PREDICTION:
        return df, "（预测功能已关闭）"

    if file_name not in TIME_SERIES_COLS or sheet_name not in TIME_SERIES_COLS[file_name]:
        return df, "（该表未配置时间序列列预测）"

    cols_to_impute = TIME_SERIES_COLS[file_name][sheet_name]
    df_imputed = df.copy()
    log_messages = []

    # 确保数据按时间和分组列排序，为插值做准备
    if time_col in df.columns:
        sort_by_cols = [group_col, time_col] if group_col in df.columns else [time_col]
        df_imputed = df_imputed.sort_values(by=sort_by_cols).reset_index(drop=True)

    for col in cols_to_impute:
        if col not in df_imputed.columns:
            continue
        original_missing = df_imputed[col].isnull().sum()
        if original_missing == 0:
            continue

        if group_col in df_imputed.columns:
            # 面板数据：按组进行插值和外推
            df_imputed[col] = df_imputed.groupby(group_col, group_keys=False)[col].apply(
                lambda s: s.interpolate(method='linear', limit_area='inside')  # 内部线性插值
                          .ffill().bfill()  # 外部用前后值填充
            )
        else:
            # 纯时间序列：整体处理
            df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_area='inside').ffill().bfill()

        new_missing = df_imputed[col].isnull().sum()
        filled_count = original_missing - new_missing
        if filled_count > 0:
            log_messages.append(f"列 '{col}'：填充了 {filled_count} 个缺失值。")
    
    log_msg = " | ".join(log_messages) if log_messages else "（无缺失值需要预测）"
    return df_imputed, log_msg

def clean_and_save_file(file_name):
    """清洗单个Excel文件并保存为CSV"""
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        print(f"[错误] 文件不存在: {file_path}")
        return False
    
    print(f"\n[处理] 正在处理文件: {file_name}")
    
    # 1. 读取Excel文件
    # 使用第一个sheet（通常名为'Sheet1'），也可以自动检测第一个工作表名
    try:
        xl = pd.ExcelFile(file_path)
        sheet_name = xl.sheet_names[0]  # 获取第一个工作表名
        df = xl.parse(sheet_name=sheet_name, dtype=object)  # 初始读取为object以保留所有信息
        print(f"  ├─ 成功读取工作表: {sheet_name}，原始形状: {df.shape}")
    except Exception as e:
        print(f"  └─ 读取文件失败: {e}")
        return False

    # 2. 移除单位行
    df = remove_unit_rows(df, file_name)
    
    # 3. 优化数据类型：自动检测数值列并转换
    for col in df.columns:
        # 尝试转换为数值，错误则强制为NaN（非数值会变成NaN）
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    # 4. 科学预测缺失值
    df_imputed, impute_log = predict_missing_values(df, file_name, sheet_name)
    
    # 5. 保存为CSV
    output_file_name = file_name.replace('.xlsx', '_cleaned.csv')
    output_path = OUTPUT_DIR / output_file_name
    try:
        df_imputed.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  ├─ 缺失值处理: {impute_log}")
        print(f"  └─ 已保存清洗后数据: {output_path} (形状: {df_imputed.shape})")
        return True
    except Exception as e:
        print(f"  └─ 保存文件失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("数据清洗脚本开始运行")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据源目录: {DATA_DIR}")
    print(f"清洗后数据将保存至: {OUTPUT_DIR}")
    print(f"缺失值预测功能: {'开启' if ENABLE_MISSING_VALUE_PREDICTION else '关闭'}")
    print("=" * 60)
    
    # 创建输出目录
    create_directories()
    
    # 处理每个文件
    success_count = 0
    for file in DATA_FILES:
        if clean_and_save_file(file):
            success_count += 1
    
    # 输出总结
    print("\n" + "=" * 60)
    print("数据清洗完成！")
    print(f"成功处理文件: {success_count}/{len(DATA_FILES)}")
    if success_count == len(DATA_FILES):
        print("所有文件已成功处理并保存至以下目录：")
        print(f"  {OUTPUT_DIR}")
        # 列出生成的文件
        print("\n生成的文件列表：")
        for f in OUTPUT_DIR.glob("*_cleaned.csv"):
            print(f"  - {f.name}")
    else:
        print(f"有 {len(DATA_FILES) - success_count} 个文件处理失败，请检查上述错误信息。")
    print("=" * 60)

# 执行脚本
if __name__ == "__main__":
    main()