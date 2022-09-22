#!/usr/bin/R

##--packages preparation
library('ggplot2')
library('tidyr')
library('dplyr')
##-environment setting
setwd("/media/wukai/Data01/RCC_classification/kidney_cancer_project/Cortical_nephrographic_training")

##--predict self testing set
##-load testset data table
cort_test <- read.csv('./Cortical/cort_model_pred_cort_set_dice_results.csv', header = TRUE, sep = ',') 
cort_test <- cort_test[,-1]
colnames(cort_test) <- c('kidney', 'tumor')
cort_sd_values <- apply(cort_test, 2, sd)
cort_mean_values <- apply(cort_test, 2, mean)

cort_test <- gather(cort_test, key = 'type', value = 'dice_coeff')
cort_test <- cbind(dataset = rep('CORT Seg Model',nrow(cort_test)), cort_test)

##-load trainset data table
nephro_test <- read.csv('./Nephro/nephro_model_pred_nephro_set_dice_results.csv', header = TRUE, sep = ',') 
nephro_test <- nephro_test[,-1]
colnames(nephro_test) <- c('kidney', 'tumor')
nephro_sd_values <- apply(nephro_test, 2, sd)
nephro_mean_values <- apply(nephro_test, 2, mean)
nephro_test <- gather(nephro_test, key = 'type', value = 'dice_coeff')
nephro_test <- cbind(dataset = rep('NEPHRO Seg Model',nrow(nephro_test)), nephro_test)

#-merge
datatable <- rbind(cort_test, nephro_test)
datatable$dataset <- factor(datatable$dataset, labels = c('CORT Seg Model','NEPHRO Seg Model'))
datatable$type <- factor(datatable$type, labels = c('kidney','tumor'))
# SD of dice coeff
mean_values <- c(cort_mean_values, nephro_mean_values)
sd_values <- c(cort_sd_values, nephro_sd_values)

dice_anno <- data.frame(dataset = c( rep('CORT Seg Model',2), rep('NEPHRO Seg Model',2)),
                        type = c('kidney','tumor','kidney','tumor'),
                        y = c(1.05,1.05,1.05,1.05),
                        dice = c('0.94 (0.06)','0.76 (0.22)','0.94 (0.04)','0.79 (0.20)'))

text_anno <- data.frame(dataset = c('CORT Seg Model','NEPHRO Seg Model'),
                        label = c('CORT Testing set', 'NEPHRO Testing set'),
                        x = c(1.5,1.5),
                        y = c(0.15, 0.15))

##--draw plot
#windowsFonts(CA=windowsFont("Calibri"))
ggplot(data = datatable, aes(x= type, y = dice_coeff))+
        geom_boxplot(aes(fill = type), width = 0.8, size = 0.7,outlier.shape = NA)+
        scale_fill_manual(values = c('#00BFC4', '#F8766D'))+#'#C23531', '#2F4554'
        geom_point(size = 2, position = position_jitter(0.2),alpha = 0.8)+
        geom_text(data = dice_anno, aes(x = type, y = y, label = dice), size = 15,show.legend = NA)+
        annotate('rect', xmin = 0.5, xmax = 2.5,ymin = 0.1,ymax = 0.2, fill = 'white',color = '#666666',size = 2)+
        geom_text(data = text_anno, aes(x = x, y = y, label = label), size = 15,color = '#666666',show.legend = NA)+
        facet_wrap(~dataset)+
        labs(y = 'dice coefficient', x = 'Segmentation Accuracy')+
        scale_x_discrete(labels= c('kidney', 'tumor'))+
        #ylim(0,16)+
        theme_bw()+
        theme(plot.title = element_text(size = 45,hjust = 0.5, face = 'bold'),
              strip.text = element_text(size =  45),
              axis.text.x = element_text(size =45),
              axis.text.y = element_text(size =45),
              axis.title.x = element_blank(),
              axis.title.y = element_text(size = 45),
              panel.grid.major.y = element_line(colour = 'gray',size = 0.7),
              legend.title = element_blank(),
              legend.text = element_text(size = 45),
              legend.background = element_blank(),
              legend.key = element_rect(fill = NA, colour = NA, size=1),
              legend.position = '')
ggsave("cort_nephro_seg_self_pred_boxplot.pdf", device = "pdf", width = 16, height = 12, dpi = 300)

##--predict each other testing set
##-load testset data table
cort_test <- read.csv('./Cortical/nephro_model_pred_cort_set_dice_results.csv', header = TRUE, sep = ',') 
cort_test <- cort_test[,-1]
colnames(cort_test) <- c('kidney', 'tumor')
cort_sd_values <- apply(cort_test, 2, sd)
cort_mean_values <- apply(cort_test, 2, mean)

cort_test <- gather(cort_test, key = 'type', value = 'dice_coeff')
cort_test <- cbind(dataset = rep('NEPHRO Seg Model',nrow(cort_test)), cort_test)

##-load trainset data table
nephro_test <- read.csv('./Nephro/cort_model_pred_nephro_set_dice_results.csv', header = TRUE, sep = ',') 
nephro_test <- nephro_test[,-1]
colnames(nephro_test) <- c('kidney', 'tumor')
nephro_sd_values <- apply(nephro_test, 2, sd)
nephro_mean_values <- apply(nephro_test, 2, mean)
nephro_test <- gather(nephro_test, key = 'type', value = 'dice_coeff')
nephro_test <- cbind(dataset = rep('CORT Seg Model',nrow(nephro_test)), nephro_test)

#-merge
datatable <- rbind(cort_test, nephro_test)
datatable$dataset <- factor(datatable$dataset, labels = c('CORT Seg Model','NEPHRO Seg Model'))
datatable$type <- factor(datatable$type, labels = c('kidney','tumor'))
# SD of dice coeff
mean_values <- c(cort_mean_values, nephro_mean_values)
sd_values <- c(cort_sd_values, nephro_sd_values)

dice_anno <- data.frame(dataset = c( rep('CORT Seg Model',2), rep('NEPHRO Seg Model',2)),
                        type = c('kidney','tumor','kidney','tumor'),
                        y = c(1.05,1.05,1.05,1.05),
                        dice = c('0.73 (0.35)','0.72 (0.28)','0.93 (0.07)','0.53 (0.33)'))

text_anno <- data.frame(dataset = c('CORT Seg Model','NEPHRO Seg Model'),
                        label = c('NEPHRO Testing set', 'CORT Testing set'),
                        x = c(1.5,1.5),
                        y = c(0.15, 0.15))

##--draw plot
#windowsFonts(CA=windowsFont("Calibri"))
ggplot(data = datatable, aes(x= type, y = dice_coeff))+
        geom_boxplot(aes(fill = type), width = 0.8, size = 0.7,outlier.shape = NA)+
        scale_fill_manual(values = c('#00BFC4', '#F8766D'))+#'#C23531', '#2F4554'
        geom_point(size = 2, position = position_jitter(0.2),alpha = 0.8)+
        geom_text(data = dice_anno, aes(x = type, y = y, label = dice), size = 15,show.legend = NA)+
        annotate('rect', xmin = 0.5, xmax = 2.5,ymin = 0.1,ymax = 0.2, fill = 'white',color = '#666666',size = 2)+
        geom_text(data = text_anno, aes(x = x, y = y, label = label), size = 15,color = '#666666',show.legend = NA)+
        facet_wrap(~dataset)+
        labs(y = 'dice coefficient', x = 'Segmentation Accuracy')+
        scale_x_discrete(labels= c('kidney', 'tumor'))+
        #ylim(0,16)+
        theme_bw()+
        theme(plot.title = element_text(size = 45,hjust = 0.5, face = 'bold'),
              strip.text = element_text(size =  45),
              axis.text.x = element_text(size =45),
              axis.text.y = element_text(size =45),
              axis.title.x = element_blank(),
              axis.title.y = element_text(size = 45),
              panel.grid.major.y = element_line(colour = 'gray',size = 0.7),
              legend.title = element_blank(),
              legend.text = element_text(size = 45),
              legend.background = element_blank(),
              legend.key = element_rect(fill = NA, colour = NA, size=1),
              legend.position = '')
ggsave("cort_nephro_seg_mutual_pred_boxplot.pdf", device = "pdf", width = 16, height = 12, dpi = 300)



