library(tidyverse)
library(gridExtra)
theme_set(theme_minimal())
library(ggthemes)

setwd("~/pmi-dependencies")

PUD_UUAS <- read_csv('R/PUD_uuas.csv')
p.pud_uuas <- PUD_UUAS %>%
  pivot_longer(cols=c(total_uuas_random_proj,
                      total_uuas_linear,
                      total_uuas_nonproj.sum),
               names_to = "dependency type",
               values_to = "uuas") %>%
  ggplot(aes(x=reorder(language,`nonproj.sum uuas / linear uuas`),
             y=uuas,
             fill=`dependency type`)) +
  geom_bar(stat="identity",
           position="dodge") +
  scale_fill_discrete(
    labels = c("connect-adjacent","bert-base-multilingual-cased","random (projective)")) +
  ggtitle("UUAS for CPMI dependencies on multilingual Parallel Universal Dependencies") +
  scale_y_continuous(limits = c(0,.7),
                     minor_breaks = seq(0, .7, 0.05),
                     breaks = seq(0, .7, 0.1)) +
  xlab("language") +
  ylab("UUAS") +
  theme(legend.position=c(.88,.85),
        axis.text.x = element_text(angle=20))

ggsave("R/plots/PUD_UUAS.pdf",
       plot=p.pud_uuas,
       width = 8.6, height = 4.0, units = "in")



