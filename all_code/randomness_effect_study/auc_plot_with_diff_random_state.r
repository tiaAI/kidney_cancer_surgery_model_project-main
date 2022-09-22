library('ggplot2')
library('dplyr')

setwd('/media/wukai/Data01/RCC_classification/kidney_cancer_project/randomness_effect_study')
#load data
datatable = read.table('randomness_check_res.txt', header = TRUE)

ggplot(data = datatable,aes(x = Random_stat, y = AUC))+
  geom_point(alpha = 0.5, size = 3,color = '#C23531',shape = 3)+
  labs(title  = 'ccRCC vs. non-ccRCC Model Performance',
       subtitle = 'with Fixed Hyperparameters',
        x = 'Random state')+
  theme_bw()+
  theme(
        plot.title = element_text(size = 50, hjust = 0.5, color = '#2F4554'),
        plot.subtitle =  element_text(size = 30, hjust = 0.5, color = 'grey'),
        axis.text.x = element_text(size =45),
        axis.text.y = element_text(size =30),
        axis.title.x = element_text(size = 45, color = '#2F4554'),
        axis.title.y = element_text(size = 45, color = '#2F4554'),
        legend.title = element_text(size = 25, color = '#2F4554', hjust = 0.5),
        legend.text = element_text(size = 25, color = '#2F4554', hjust = 0.5),
        legend.background = element_rect(fill = NA, color = '#2F4554', size = 1.2),
        legend.key = element_rect(fill = NA, colour = NA))
ggsave('auc_randomness_check.jpg', device = 'jpeg', width = 16, height = 12, dpi = 300)
