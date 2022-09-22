library('ggplot2')
library('tidyr')
library('dplyr')
library('tibble')
library('pROC')
library('latex2exp')

setwd("/media/wukai/Data01/RCC_classification/kidney_cancer_project/Cortical_nephrographic_training")
options(digits=4)

###---1. ccRCC vs. non-ccRCC in CORT Classfication Model
# for ccRCC vs. non-ccRCC prediction with CORT pred CORT testing set
# load data
dataset1 <- read.csv('subtype_proba_cort_pred_cort.csv', header = TRUE, sep = ',')
rocobj1 <- pROC::roc(dataset1$truth, dataset1$proba)
auc_ci1 <- as.numeric(ci(rocobj1))
roc_table1 <- data.frame(FPR = 1-rocobj1$specificities, TPR = rocobj1$sensitivities)
roc_table1 <- roc_table1[order(roc_table1$TPR),]

# for ccRCC vs. non-ccRCC prediction with CORT pred NEPHRO testing set
# load data
dataset2 <- read.csv('subtype_proba_cort_pred_nephro.csv', header = TRUE, sep = ',')
rocobj2 <- pROC::roc(dataset2$truth, dataset2$proba)
auc_ci2 <- as.numeric(ci(rocobj2))
roc_table2 <- data.frame(FPR = 1-rocobj2$specificities, TPR = rocobj2$sensitivities)
roc_table2 <- roc_table2[order(roc_table2$TPR),]

roc.test(rocobj1, rocobj2, method="bootstrap")
# 
box <- data.frame(x = c(0.35,0.35,1.01,1.01),
                  y = c(0.01,0.32,0.32,0.01))
line_table1 <- data.frame(x= c(0.4, 0.44),
                         y = c(0.18,0.18))
line_table2 <- data.frame(x= c(0.4, 0.44),
                          y = c(0.1,0.1))

ggplot()+
  geom_line(data = roc_table1, aes(x = FPR, y = TPR), colour = '#D48265', size = 3.5, alpha = 0.6)+
  geom_line(data = roc_table2, aes(x = FPR, y = TPR), colour = '#61A0A8', size = 3.5, alpha = 0.6)+
  geom_abline(slope = 1, intercept =0, linetype = 3, size = 0.7, show.legend = NA)+
  geom_polygon(data = box, aes(x=x , y=y), fill = 'white', color = '#666666',size = 1.5)+
  annotate('text', x =  0.20, y =0.9, label =  'CORT Testing set', color =  '#D48265', size = 15)+
  annotate('text', x =  0.7, y =0.6, label =  'NEPHRO Testing set', color =  '#61A0A8', size = 15)+
  annotate('text', x =  0.70, y =0.27, label =  'Bootstrap test of Two ROC (P = 0.3)', size = 12, color = '#666666')+
  annotate('text', x =  0.70, y =0.18, label =  TeX('ROC (AUC = 0.80$\\pm$0.15)'), size = 12)+
  annotate('text', x =  0.70, y =0.1, label =  TeX('ROC (AUC = 0.67$\\pm$0.15)'), size = 12)+

  geom_line(data = line_table1, aes(x = x, y = y), colour = '#D48265', size = 3.5, alpha = 0.6)+
  geom_line(data = line_table2, aes(x = x, y = y), colour = '#61A0A8', size = 3.5, alpha = 0.6)+  
  
  labs(title = 'CORT Classification Model',
       subtitle = 'ccRCC vs. non-ccRCC (resampling)',
       y = 'Sensitivity',
       x = '1 - specificity')+
  theme_bw()+
  theme(plot.title = element_text(size = 55,hjust = 0.5),
        plot.subtitle = element_text(size = 45,hjust = 0.5, color = '#666666'),
        axis.text.x = element_text(size =45),
        axis.text.y = element_text(size =45),
        axis.title.x = element_text(size = 45),
        axis.title.y = element_text(size = 45),
        legend.title = element_blank(),
        legend.text = element_text(size = 45),
        legend.background = element_blank(),
        legend.key = element_rect(fill = NA, colour = NA, size=1),
        legend.position = '')

ggsave("subtype_roc_cort_model_resampled.pdf", device = "pdf", width = 16, height = 12, dpi = 300)

###########################################

###---2. ccRCC vs. non-ccRCC in NEPHRO Classfication Model
# for ccRCC vs. non-ccRCC prediction with NEPHRO pred CORT testing set
# load data
dataset1 <- read.csv('subtype_proba_nephro_pred_cort.csv', header = TRUE, sep = ',')
rocobj1 <- pROC::roc(dataset1$truth, dataset1$proba)
auc_ci1 <- as.numeric(ci(rocobj1))
roc_table1 <- data.frame(FPR = 1-rocobj1$specificities, TPR = rocobj1$sensitivities)
roc_table1 <- roc_table1[order(roc_table1$TPR),]

# for ccRCC vs. non-ccRCC prediction with  NEPHRO  pred NEPHRO testing set
# load data
dataset2 <- read.csv('subtype_proba_nephro_pred_nephro.csv', header = TRUE, sep = ',')
rocobj2 <- pROC::roc(dataset2$truth, dataset2$proba)
auc_ci2 <- as.numeric(ci(rocobj2))
roc_table2 <- data.frame(FPR = 1-rocobj2$specificities, TPR = rocobj2$sensitivities)
roc_table2 <- roc_table2[order(roc_table2$TPR),]

roc.test(rocobj1, rocobj2, method="bootstrap")
# 
box <- data.frame(x = c(0.35,0.35,1.02,1.02),
                  y = c(0.01,0.32,0.32,0.01))
line_table1 <- data.frame(x= c(0.4, 0.44),
                          y = c(0.18,0.18))
line_table2 <- data.frame(x= c(0.4, 0.44),
                          y = c(0.1,0.1))

ggplot()+
  geom_line(data = roc_table1, aes(x = FPR, y = TPR), colour = '#D48265', size = 3.5, alpha = 0.6)+
  geom_line(data = roc_table2, aes(x = FPR, y = TPR), colour = '#61A0A8', size = 3.5, alpha = 0.6)+
  geom_abline(slope = 1, intercept =0, linetype = 3, size = 0.7, show.legend = NA)+
  geom_polygon(data = box, aes(x=x , y=y), fill = 'white', color = '#666666',size = 1.5)+
  annotate('text', x =  0.20, y =0.95, label =  'NEPHRO Testing set', color = '#61A0A8' , size = 15)+
  annotate('text', x =  0.7, y =0.6, label =  'CORT Testing set', color =  '#D48265', size = 15)+
  annotate('text', x =  0.70, y =0.27, label =  'Bootstrap test of Two ROC (P = 0.02)', size = 12, color = '#666666')+
  annotate('text', x =  0.70, y =0.18, label =  TeX('ROC (AUC = 0.89$\\pm$0.06)'), size = 12)+
  annotate('text', x =  0.70, y =0.1, label =  TeX('ROC (AUC = 0.66$\\pm$0.18)'), size = 12)+
  
  geom_line(data = line_table1, aes(x = x, y = y), colour ='#61A0A8', size = 3.5, alpha = 0.6)+
  geom_line(data = line_table2, aes(x = x, y = y), colour =  '#D48265', size = 3.5, alpha = 0.6)+  
  
  labs(title = 'NEPHRO Classification Model',
       subtitle = 'ccRCC vs. non-ccRCC (resampling)',
       y = 'Sensitivity',
       x = '1 - specificity')+
  theme_bw()+
  theme(plot.title = element_text(size = 55,hjust = 0.5),
        plot.subtitle = element_text(size = 45,hjust = 0.5, color = '#666666'),
        axis.text.x = element_text(size =45),
        axis.text.y = element_text(size =45),
        axis.title.x = element_text(size = 45),
        axis.title.y = element_text(size = 45),
        legend.title = element_blank(),
        legend.text = element_text(size = 45),
        legend.background = element_blank(),
        legend.key = element_rect(fill = NA, colour = NA, size=1),
        legend.position = '')

ggsave("subtype_roc_nephro_model_resampled.pdf", device = "pdf", width = 16, height = 12, dpi = 300)

###########################################
###---3. T1&T2 vs. T3&T4 in CORT Classfication Model
# for T1&T2 vs. T3&T4  prediction with CORT pred CORT testing set
# load data
dataset1 <- read.csv('stage_proba_cort_pred_cort.csv', header = TRUE, sep = ',')
rocobj1 <- pROC::roc(dataset1$truth, dataset1$proba)
auc_ci1 <- as.numeric(ci(rocobj1))
roc_table1 <- data.frame(FPR = 1-rocobj1$specificities, TPR = rocobj1$sensitivities)
roc_table1 <- roc_table1[order(roc_table1$TPR),]

# for T1&T2 vs. T3&T4  prediction with CORT pred NEPHRO testing set
# load data
dataset2 <- read.csv('stage_proba_cort_pred_nephro.csv', header = TRUE, sep = ',')
rocobj2 <- pROC::roc(dataset2$truth, dataset2$proba)
auc_ci2 <- as.numeric(ci(rocobj2))
roc_table2 <- data.frame(FPR = 1-rocobj2$specificities, TPR = rocobj2$sensitivities)
roc_table2 <- roc_table2[order(roc_table2$TPR),]

roc.test(rocobj1, rocobj2, method="bootstrap")
# 
box <- data.frame(x = c(0.35,0.35,1.02,1.02),
                  y = c(0.01,0.32,0.32,0.01))
line_table1 <- data.frame(x= c(0.4, 0.44),
                          y = c(0.18,0.18))
line_table2 <- data.frame(x= c(0.4, 0.44),
                          y = c(0.1,0.1))

ggplot()+
  geom_line(data = roc_table1, aes(x = FPR, y = TPR), colour = '#D48265', size = 3.5, alpha = 0.6)+
  geom_line(data = roc_table2, aes(x = FPR, y = TPR), colour = '#61A0A8', size = 3.5, alpha = 0.6)+
  geom_abline(slope = 1, intercept =0, linetype = 3, size = 0.7, show.legend = NA)+
  geom_polygon(data = box, aes(x=x , y=y), fill = 'white', color = '#666666',size = 1.5)+
  annotate('text', x =  0.7, y =0.6,  label =  'NEPHRO Testing set', color = '#61A0A8' , size = 15)+
  annotate('text',x =  0.20, y =0.95, label =  'CORT Testing set', color =  '#D48265', size = 15)+
  annotate('text', x =  0.70, y =0.27, label =  'Bootstrap test of Two ROC (P = 0.2)', size = 12, color = '#666666')+
  annotate('text', x =  0.70, y =0.18, label =  TeX('ROC (AUC = 0.75$\\pm$0.11)'), size = 12)+
  annotate('text', x =  0.70, y =0.1, label =  TeX('ROC (AUC = 0.85$\\pm$0.09)'), size = 12)+
  
  geom_line(data = line_table1, aes(x = x, y = y), colour ='#61A0A8', size = 3.5, alpha = 0.6)+
  geom_line(data = line_table2, aes(x = x, y = y), colour =  '#D48265', size = 3.5, alpha = 0.6)+  
  
  labs(title = 'CORT Classification Model',
       subtitle = ' T1&T2 vs. T3&T4 (resampling)',
       y = 'Sensitivity',
       x = '1 - specificity')+
  theme_bw()+
  theme(plot.title = element_text(size = 55,hjust = 0.5),
        plot.subtitle = element_text(size = 45,hjust = 0.5, color = '#666666'),
        axis.text.x = element_text(size =45),
        axis.text.y = element_text(size =45),
        axis.title.x = element_text(size = 45),
        axis.title.y = element_text(size = 45),
        legend.title = element_blank(),
        legend.text = element_text(size = 45),
        legend.background = element_blank(),
        legend.key = element_rect(fill = NA, colour = NA, size=1),
        legend.position = '')

ggsave("stage_roc_cort_model_resampled.pdf", device = "pdf", width = 16, height = 12, dpi = 300)


###########################################
###---4. T1&T2 vs. T3&T4 in NEPHRO Classfication Model
# for T1&T2 vs. T3&T4  prediction with NEPHRO pred CORT testing set
# load data
dataset1 <- read.csv('stage_proba_nephro_pred_cort.csv', header = TRUE, sep = ',')
rocobj1 <- pROC::roc(dataset1$truth, dataset1$proba)
auc_ci1 <- as.numeric(ci(rocobj1))
roc_table1 <- data.frame(FPR = 1-rocobj1$specificities, TPR = rocobj1$sensitivities)
roc_table1 <- roc_table1[order(roc_table1$TPR),]

# for  T1&T2 vs. T3&T4  prediction with NEPHRO pred NEPHRO testing set
# load data
dataset2 <- read.csv('stage_proba_nephro_pred_nephro.csv', header = TRUE, sep = ',')
rocobj2 <- pROC::roc(dataset2$truth, dataset2$proba)
auc_ci2 <- as.numeric(ci(rocobj2))
roc_table2 <- data.frame(FPR = 1-rocobj2$specificities, TPR = rocobj2$sensitivities)
roc_table2 <- roc_table2[order(roc_table2$TPR),]

roc.test(rocobj1, rocobj2, method="bootstrap")
# 
box <- data.frame(x = c(0.35,0.35,1.02,1.02),
                  y = c(0.01,0.32,0.32,0.01))
line_table1 <- data.frame(x= c(0.4, 0.44),
                          y = c(0.18,0.18))
line_table2 <- data.frame(x= c(0.4, 0.44),
                          y = c(0.1,0.1))

ggplot()+
  geom_line(data = roc_table1, aes(x = FPR, y = TPR), colour = '#D48265', size = 3.5, alpha = 0.6)+
  geom_line(data = roc_table2, aes(x = FPR, y = TPR), colour = '#61A0A8', size = 3.5, alpha = 0.6)+
  geom_abline(slope = 1, intercept =0, linetype = 3, size = 0.7, show.legend = NA)+
  geom_polygon(data = box, aes(x=x , y=y), fill = 'white', color = '#666666',size = 1.5)+
  annotate('text', x =  0.20, y =0.95,  label =  'NEPHRO Testing set', color = '#61A0A8' , size = 15)+
  annotate('text', x =  0.8, y =0.47, label =  'CORT Testing set', color =  '#D48265', size = 15)+
  annotate('text', x =  0.70, y =0.27, label =  'Bootstrap test of Two ROC (P = 0.008)', size = 12, color = '#666666')+
  annotate('text', x =  0.70, y =0.18, label =  TeX('ROC (AUC = 0.772$\\pm$0.1)'), size = 12)+
  annotate('text', x =  0.70, y =0.1, label =  TeX('ROC (AUC = 0.530$\\pm$0.15)'), size = 12)+
  
  geom_line(data = line_table1, aes(x = x, y = y), colour ='#61A0A8', size = 3.5, alpha = 0.6)+
  geom_line(data = line_table2, aes(x = x, y = y), colour =  '#D48265', size = 3.5, alpha = 0.6)+  
  
  labs(title = 'NEPHRO Classification Model',
       subtitle = ' T1&T2 vs. T3&T4 (resampling)',
       y = 'Sensitivity',
       x = '1 - specificity')+
  theme_bw()+
  theme(plot.title = element_text(size = 55,hjust = 0.5),
        plot.subtitle = element_text(size = 45,hjust = 0.5, color = '#666666'),
        axis.text.x = element_text(size =45),
        axis.text.y = element_text(size =45),
        axis.title.x = element_text(size = 45),
        axis.title.y = element_text(size = 45),
        legend.title = element_blank(),
        legend.text = element_text(size = 45),
        legend.background = element_blank(),
        legend.key = element_rect(fill = NA, colour = NA, size=1),
        legend.position = '')

ggsave("stage_roc_nephro_model_resampled.pdf", device = "pdf", width = 16, height = 12, dpi = 300)


