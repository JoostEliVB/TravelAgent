import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import wilcoxon, shapiro

# 读取数据
file_path = "Chatbot_Evaluation_Numeric.xlsx"
df = pd.read_excel(file_path)

# 创建一个字典，用于存储分组结果：键为问题的基础名称，值为对应的列名列表
grouped_questions = defaultdict(list)
for col in df.columns:
    # 如果列名以 ".2" 结尾，则去除后缀
    if col.endswith(".2"):
        base = col[:-2].strip()
    else:
        base = col.strip()
    # 移除末尾的句号（如果有的话）
    base = base.rstrip('.')
    grouped_questions[base].append(col)

# 打印出每个问题对应的列名
for question, cols in grouped_questions.items():
    print(f"问题: '{question}' -> 列名: {cols}")

results = []
for question, cols in grouped_questions.items():
    # Only process groups with exactly two columns (pre and post)
    if len(cols) == 2:
        print(f"问题: '{question}' -> 列名: {cols}")
        # We assume the first column is pre-LTM and the second is post-LTM.
        pre_col, post_col = cols[0], cols[1]
        data = df[[pre_col, post_col]].dropna()
        N = len(data)
        if N < 1:
            continue  # Skip if no paired data
        
        pre_vals = data[pre_col]
        post_vals = data[post_col]
        
        # Perform Shapiro-Wilk test for normality
        _, pre_shapiro_p = shapiro(pre_vals)
        _, post_shapiro_p = shapiro(post_vals)
        pre_normal = pre_shapiro_p > 0.05
        post_normal = post_shapiro_p > 0.05
        both_normal = pre_normal and post_normal
        
        # Calculate means
        pre_mean = pre_vals.mean()
        post_mean = post_vals.mean()
        
        # Perform the Wilcoxon signed-rank test using the library
        try:
            W, p_val = wilcoxon(post_vals, pre_vals)
        except Exception as e:
            print(f"Error processing {question}: {str(e)}")
            W, p_val = np.nan, np.nan
        
        # Calculate effect size only if we have valid W and p_val
        if not np.isnan(W) and not np.isnan(p_val):
            # Compute expected mean and standard deviation for the Wilcoxon W statistic under H0:
            mu_W = N * (N + 1) / 4
            sigma_W = np.sqrt(N * (N + 1) * (2 * N + 1) / 24)
            # Compute Z-value based on the test statistic (W)
            z = (W - mu_W) / sigma_W
            effect_size = abs(z) / np.sqrt(N)
        else:
            effect_size = np.nan
        
        results.append({
            "Question": question,
            "Pre Column": pre_col,
            "Post Column": post_col,
            "N": N,
            "Pre Mean": round(pre_mean, 2),
            "Post Mean": round(post_mean, 2),
            "Mean Difference": round(post_mean - pre_mean, 2),
            "Pre Shapiro-Wilk p": round(pre_shapiro_p, 2),
            "Post Shapiro-Wilk p": round(post_shapiro_p, 2),
            "Normally Distributed": "Yes" if both_normal else "No",
            "Wilcoxon W": round(W, 2) if not np.isnan(W) else W,
            "p-value": round(p_val, 2) if not np.isnan(p_val) else p_val,
            "Effect Size (r)": round(effect_size, 2) if not np.isnan(effect_size) else effect_size
        })

# Create a DataFrame to display the results
results_df = pd.DataFrame(results)

# Set display options to show rounded numbers
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("\nWilcoxon Test Results with Effect Size and Means:")
print(results_df)

# Save results to Excel file with rounded numbers
output_file = "wilcoxon_results.xlsx"
results_df.to_excel(output_file, index=False, float_format="%.2f")
print(f"\nResults have been saved to {output_file}")