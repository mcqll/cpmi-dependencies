library(tidyverse)
theme_set(theme_minimal())
#setwd("/Users/j/McGill/PhD-miasma/pmi-dependencies/R")

lstmacc <- read_csv("lstmresults.csv", col_names = TRUE)
lstmacc$model <- factor(lstmacc$model)
library(plyr)
lstmacc$model <- mapvalues(lstmacc$model, from = c("lstm", "onlstm", "onlstm_syd"), to = c("LSTM", "ONLSTM", "ONSLTM-SYD"))
baselines <- data.frame(algorithm = c("nonproj", "proj"), uuas = c(.13, .26))
lstmacc %>%
  ggplot(aes(x=absolute_value, y=uuas, colour=model, shape=model)) +
  facet_grid(~algorithm) +
  geom_point() +
  geom_hline(data=baselines, aes(yintercept=uuas), linetype="dashed", color = "grey") +
  geom_hline(yintercept=.5, linetype="dashed", color = "grey") +
  ylab("Accuracy (UUAS)") + xlab("with absolute value")

lstmacc %>% filter(absolute_value == T) %>%
  ggplot(aes(x=model, y=uuas, colour=algorithm)) +
  geom_point() +
  ggtitle("Accuracy of LSTM models") +
  ylab("Accuracy (UUAS)") + xlab("Model")
