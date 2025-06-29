library(reticulate)
library(anndata)
library(githubinstall)
library(devtools)
library(Seurat)

library('SPARK')
library(Matrix)
library(reticulate)


# 导入 anndata 包
ad <- import("anndata", convert = FALSE)
np <- import("numpy")


library(SPARK)
library(dplyr)

# === 1. 加载数据 ===
d12 <- read.csv("F:\\WiK\\data/mouse_brain/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate2_cell_by_gene_S1R2.csv", row.names = 1)
d12_meta <- read.csv("F:\\WiK\\data/mouse_brain/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate2_cell_metadata_S1R2.csv", row.names = 1)

# 确保行名对齐
if (!all(rownames(d12) == rownames(d12_meta))) {
  stop("d12 和 d12_meta 的行名不一致，请检查数据！")
}

# === 2. 转换坐标并筛选点位 ===
# 添加筛选规则：d12_right 筛选条件
d12_meta$coord_filter <- with(d12_meta, center_x * 6 / 11 + 2436.36 - center_y > 0)

# 仅保留满足条件的点位
d12_right_meta <- d12_meta[d12_meta$coord_filter, ]
d12_right <- d12[rownames(d12_right_meta), ]

# === 3. 构造 rawcount 和 info 数据 ===
# 构造列名：四舍五入并拼接成 "x_y" 格式
coords <- paste(
  round(d12_right_meta$center_x, 3),
  round(d12_right_meta$center_y, 3),
  sep = "x"
)

# 转置表达矩阵：行 = 基因，列 = 坐标点
rawcount <- t(d12_right)
colnames(rawcount) <- coords  # 使用格式化后的坐标名

# 构造 info 数据框：记录坐标和总表达量
info <- data.frame(
  x = round(d12_right_meta$center_x, 3),
  y = round(d12_right_meta$center_y, 3),
  total_counts = colSums(rawcount)
)
rownames(info) <- coords

# === 4. 创建 SPARK 对象并运行分析 ===
# 创建 SPARK 对象
spark <- CreateSPARKObject(
  counts = rawcount,  # 基因 × 坐标点
  location = info[, 1:2],  # 仅需要 x 和 y 坐标
  percentage = 0.1,
  min_total_counts = 10
)

# 添加库尺寸信息
spark@lib_size <- apply(spark@counts, 2, sum)

# 拟合方差成分模型
spark <- spark.vc(
  spark, 
  covariates = NULL, 
  lib_size = spark@lib_size, 
  num_core = 4, 
  verbose = FALSE
)

# 运行空间基因表达检验
spark <- spark.test(
  spark, 
  check_positive = TRUE, 
  verbose = FALSE
)

# === 5. 查看结果 ===
cat("SPARK 分析结果：\n")
head(spark@res_mtest[, c("combined_pvalue", "adjusted_pvalue")])




#---------------------------------------------------------------------------减小维度

# 降低点位数量
set.seed(42)
sample_size <- 5000
sampled_cols <- sample(colnames(rawcount), size = sample_size)

rawcount_sampled <- rawcount[, sampled_cols]
info_sampled <- info[sampled_cols, ]

# 重新创建 SPARK 对象
spark <- CreateSPARKObject(
  counts = rawcount_sampled,
  location = info_sampled[, 1:2],
  percentage = 0.1,
  min_total_counts = 10
)
spark@lib_size <- apply(spark@counts, 2, sum)

# 筛选高变异基因
top_genes <- names(sort(apply(rawcount_sampled, 1, var), decreasing = TRUE)[1:200])
rawcount_top <- rawcount_sampled[top_genes, ]

# 更新 SPARK 对象
spark <- CreateSPARKObject(
  counts = rawcount_top,
  location = info_sampled[, 1:2],
  percentage = 0.1,
  min_total_counts = 10
)
spark@lib_size <- apply(spark@counts, 2, sum)

# 运行 VC 和测试
spark <- spark.vc(
  spark, 
  covariates = NULL, 
  lib_size = spark@lib_size, 
  num_core = 2, 
  verbose = TRUE,
)

spark <- spark.test(
  spark, 
  check_positive = TRUE, 
  verbose = FALSE
)

# 查看结果
head(spark@res_mtest[, c("combined_pvalue", "adjusted_pvalue")])

# 检查这些基因是否存在，并提取它们的 p 值
available_genes <- genes[genes %in% rownames(spark@res_mtest)]
missing_genes <- genes[!genes %in% rownames(spark@res_mtest)]

# 如果存在基因，提取它们的 p 值
if (length(available_genes) > 0) {
  p_values <- spark@res_mtest[available_genes, c("combined_pvalue", "adjusted_pvalue")]
  print(p_values)
} else {
  cat("没有找到任何指定的基因。\n")
}

# 显示未找到的基因
if (length(missing_genes) > 0) {
  cat("以下基因不在结果列表中：", paste(missing_genes, collapse = ", "), "\n")
}

top_genes200 <- head(rownames(spark@res_mtest[order(spark@res_mtest$combined_pvalue), ]), 200)

top_genes91 <- head(rownames(spark@res_mtest[order(spark@res_mtest$combined_pvalue), ]), 91)

top_genes649 <- head(rownames(spark@res_mtest[order(spark@res_mtest$combined_pvalue), ]), 649)


top_genes_with_pvalues1000 <- spark@res_mtest[order(spark@res_mtest$combined_pvalue), ][1:1000, c("combined_pvalue", "adjusted_pvalue")]

# 输出前 100 个基因及其 p 值
print(top_genes_with_pvalues)

write.csv(top_genes649, file = "F:\\WiK\\R\\Merfish649genes.csv", row.names = TRUE)
