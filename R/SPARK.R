install.packages("devtools")
install.packages("githubinstall")
install.packages("anndata")
install.packages("Seurat")
install.packages("reticulate")
library(reticulate)
library(anndata)
library(githubinstall)
Sys.setenv(PKG_BUILD_VIGNETTES=TRUE)
library(devtools)
install_github("xzhoulab/SPARK")

library(Seurat)
devtools::install_local('C:\\Users\\woshi\\Downloads\\SPARK-master.zip')

library('SPARK')

load("C:\\Users\\woshi\\Downloads\\Layer2_BC_Count.rds")

rawcount[1:5,1:5]

colnames(rawcount)
info <- cbind.data.frame(x=as.numeric(sapply(strsplit(colnames(rawcount),split="x"),"[",1)),
                         y=as.numeric(sapply(strsplit(colnames(rawcount),split="x"),"[",2)),
                         total_counts=apply(rawcount,2,sum))

rownames(info) <- colnames(rawcount)

spark <- CreateSPARKObject(counts=rawcount, 
                           location=info[,1:2],
                           percentage = 0.1, 
                           min_total_counts = 10)


# 导入 anndata 包
ad <- import("anndata", convert = FALSE)

# 读取 .h5ad 文件
adata151676 <- ad$read_h5ad("F:\\WiK\\data\\LIBD\\data_151676.h5ad")



# 提取表达矩阵并转换为 R 矩阵
rawcount <- as.matrix(py_to_r(adata151676$X))

# 提取坐标信息并转换为 R 数据框
obs_data <- py_to_r(adata151676$obs)
info <- data.frame(
  x2 = obs_data[["x2"]],
  x3 = obs_data[["x3"]]
)

#--------------------------------


# Step 1: 从 adata151676 提取坐标信息和总计数
# 使用 x2 和 x3 列作为坐标，使用 n_counts 作为总表达量
location_info <- cbind.data.frame(
  x = as.numeric(adata151676$obs[["x2"]]),
  y = as.numeric(adata151676$obs[["x3"]]),
  total_counts = as.numeric(adata151676$obs[["n_counts"]])
)
rownames(location_info) <- rownames(adata151676$obs)  # 行名设为细胞ID

# Step 2: 提取 adata151676 的表达矩阵 (基因 x 细胞)
expression_matrix <- t(as.matrix(adata151676$X))  # 转置为基因 x 点位格式
colnames(expression_matrix) <- rownames(location_info)  # 设置列名为 location 的行名

# Step 3: 使用 SPARK 创建对象
spark <- CreateSPARKObject(
  counts = expression_matrix,      # 基因 x 点位 矩阵
  location = location_info,        # 包含 x, y 和 total_counts 的坐标数据
  percentage = 0.1,
  min_total_counts = 10
)

# 检查 SPARK 对象的库大小
spark@lib_size <- rowSums(spark@counts)
#==================================================================


# Step 4: 拟合统计模型
if (ncol(expression_matrix) != nrow(location_info)) {
  stop("表达矩阵的列数和位置数据的行数不匹配！")
}

# 重新计算 lib_size，确保与细胞数量匹配
spark@lib_size <- colSums(spark@counts)  # 使用列求和，确保是每个细胞的总表达量

# 继续拟合模型
spark <- spark.vc(
  spark, 
  covariates = NULL, 
  lib_size = spark@lib_size, 
  num_core = 5, 
  verbose = FALSE
)

# 计算空间表达模式基因的 p 值
spark <- spark.test(
  spark, 
  check_positive = TRUE, 
  verbose = FALSE
)

# 查看结果
head(spark@res_mtest[, c("combined_pvalue", "adjusted_pvalue")])
#三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三
genes <- c('SNAP25', 'CAMK2N1', 'KRT17', 'CNR1', 'SCGB2A2', 'SCD', 'FAU',
               'TMEM144', 'SEMA3E', 'YWHAG', 'RPS28', 'SCGB1D2', 'GPX4', 'NCS1',
               'RPL8', 'TP53INP2', 'CPNE5', 'SNCA', 'PGAM1', 'CLU', 'PLXNB1', 'TRIO',
               'TUBB2A', 'HS3ST2', 'RPS27A', 'TSC22D4', 'SSTR2', 'HBA2', 'MAP1B',
               'ISG15', 'KRT19', 'GAD1', 'C4ORF48', 'CHD3', 'ATP1A3', 'ARPP19',
               'CDK14', 'MUC1', 'MAP1LC3A', 'PWAR6', 'PHLDB1', 'PIANP', 'RPL37',
               'CD74', 'CALM1', 'BCL11A', 'MAP2K1', 'SYN2', 'NEUROD2', 'GFAP')



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

top_genes1000 <- head(rownames(spark@res_mtest[order(spark@res_mtest$combined_pvalue), ]), 1000)



top_genes_with_pvalues1000 <- spark@res_mtest[order(spark@res_mtest$combined_pvalue), ][1:1000, c("combined_pvalue", "adjusted_pvalue")]

# 输出前 100 个基因及其 p 值
print(top_genes_with_pvalues)

top_genes200
write.csv(top_genes200, file = "F:\\WiK\\R\\top200genes.csv", row.names = TRUE)

write.csv(top_genes_with_pvalues1000, file = "F:\\WiK\\R\\top1000genes&Value.csv", row.names = TRUE)


