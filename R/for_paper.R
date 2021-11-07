library(tidyverse)
library(gridExtra)
theme_set(theme_minimal())
library(ggthemes)
library(scales)

#### LOAD MODELS ####

USE_ABSOLUTE_VALUE <- F
USE_WSJ10 <- F

results_dir <- "results/results-clean"
csv_file_loc <- "wordpair*.csv" # for non-absolute value

if (USE_ABSOLUTE_VALUE) {csv_file_loc <- "loaded*/wordpair*.csv"}
if (USE_WSJ10) {results_dir <- "results/results-comparison-wsj10"}

bart <- read_csv(Sys.glob(file.path(
  results_dir, "cems/bart-large*", csv_file_loc)))
bert_base <- read_csv(Sys.glob(file.path(
  results_dir, "cems/bert-base-cased*", csv_file_loc)))
bert_large <- read_csv(Sys.glob(file.path(
  results_dir, "cems/bert-large-cased*", csv_file_loc)))
dbert <- read_csv(Sys.glob(file.path(
  results_dir, "cems/distilbert-base-cased*", csv_file_loc)))
gpt2 <- read_csv(Sys.glob(file.path(
  results_dir, "cems/gpt2*", csv_file_loc)))
w2v <- read_csv(Sys.glob(file.path(
  results_dir, "cems/w2v*", csv_file_loc)))
xlm <- read_csv(Sys.glob(file.path(
  results_dir, "cems/xlm-mlm-en-2048*", csv_file_loc)))
xlnet_base <- read_csv(Sys.glob(file.path(
  results_dir, "cems/xlnet-base-cased*", csv_file_loc)))
xlnet_large <- read_csv(Sys.glob(file.path(
  results_dir, "cems/xlnet-large-cased*", csv_file_loc)))

lstm <- read_csv(Sys.glob(file.path(
  results_dir, "lstms/loaded=lstm_pad0*", csv_file_loc)))
onlstm <- read_csv(Sys.glob(file.path(
  results_dir, "lstms/loaded=onlstm_pad0*", csv_file_loc)))
onlstm_syd <- read_csv(Sys.glob(file.path(
  results_dir, "lstms/loaded=onlstm_syd_pad0*", csv_file_loc)))

baseline_linear <- read_csv(Sys.glob(file.path(
  results_dir, "baselines/linear_baseline*", "wordpair*.csv")))
baseline_random <- read_csv(Sys.glob(file.path(
  results_dir, "baselines/random_baseline_pad0_1_*", "wordpair*.csv")))
baseline_random2 <- read_csv(Sys.glob(file.path(
  results_dir, "baselines/random_baseline_pad0_2_*", "wordpair*.csv")))

bart$model  <- "Bart"
bert_base$model  <- "BERT-base"
bert_large$model  <- "BERT-large"
dbert$model <- "DistilBERT"
gpt2$model  <- "GPT2"
w2v$model   <- "Word2Vec"
xlm$model   <- "XLM"
xlnet_large$model <- "XLNet-large"
xlnet_base$model <- "XLNet-base"

lstm$model <- "LSTM"
onlstm$model <- "ONLSTM"
onlstm_syd$model <- "ONLSTM-SYD"

baseline_linear$model <- "baseline_linear"
baseline_random$model <- "baseline_random"
baseline_random2$model <- "baseline_random2"

all_models_raw <- list(
  dbert,
  bert_base,
  bert_large,
  xlnet_base,
  xlnet_large,
  xlm,
  bart,
  gpt2,
  w2v,
  lstm,
  onlstm,
  onlstm_syd,
  baseline_linear,
  baseline_random,
  baseline_random2)

results_dir_2020 <- "results/results-clean-2020"

pos_simple_bert_base <- read_csv(Sys.glob(file.path(
  results_dir_2020, "pos-cpmi/simple_probe/xpos_bert-base*", csv_file_loc)))
pos_simple_bert_large <- read_csv(Sys.glob(file.path(
  results_dir_2020, "pos-cpmi/simple_probe/xpos_bert-large*", csv_file_loc)))
pos_simple_xlnet_base <- read_csv(Sys.glob(file.path(
  results_dir_2020, "pos-cpmi/simple_probe/xpos_xlnet-base*", csv_file_loc)))
pos_simple_xlnet_large <- read_csv(Sys.glob(file.path(
  results_dir_2020, "pos-cpmi/simple_probe/xpos_xlnet-large*", csv_file_loc)))

pos_ib_bert_base <- read_csv(Sys.glob(file.path(
  results_dir_2020, "pos-cpmi/IB_probe/IB_xpos_bert-base*", csv_file_loc)))
pos_ib_bert_large <- read_csv(Sys.glob(file.path(
  results_dir_2020, "pos-cpmi/IB_probe/IB_xpos_bert-large*", csv_file_loc)))
pos_ib_xlnet_base <- read_csv(Sys.glob(file.path(
  results_dir_2020, "pos-cpmi/IB_probe/IB_xpos_xlnet-base*", csv_file_loc)))
pos_ib_xlnet_large <- read_csv(Sys.glob(file.path(
  results_dir_2020, "pos-cpmi/IB_probe/IB_xpos_xlnet-large*", csv_file_loc)))

pos_simple_bert_base$model <-  "XPOS_simple_BERTbase"
pos_simple_bert_large$model <- "XPOS_simple_BERTlarge"
pos_simple_xlnet_base$model <- "XPOS_simple_XLNetbase"
pos_simple_xlnet_large$model <-"XPOS_simple_XLNetlarge"
pos_ib_bert_base$model <-  "XPOS_IB_BERTbase"
pos_ib_bert_large$model <- "XPOS_IB_BERTlarge"
pos_ib_xlnet_base$model <- "XPOS_IB_XLNetbase"
pos_ib_xlnet_large$model <-"XPOS_IB_XLNetlarge"


all_pos_models_raw <- list(pos_simple_bert_base,
                      pos_simple_bert_large,
                      pos_simple_xlnet_base,
                      pos_simple_xlnet_large,
                      pos_ib_bert_base,
                      pos_ib_bert_large,
                      pos_ib_xlnet_base,
                      pos_ib_xlnet_large)

make_dep_len_nopunct <- function(dataframe, verbose=TRUE) {
  # makes a dep_len feature, which records the distance ignoring tokens not in
  # gold tree that is, skipping over punctuation or other ignored symbols in
  # calculating distance.
  if (verbose) {
    message(dataframe$model[[1]])
    }
  newdf <- tibble()
  # dataframe = dataframe %>% filter(sentence_index %in% c(1,2,3,4))
  pb <- txtProgressBar(style = 3)
  for (s_index in unique(dataframe$sentence_index)) {
    # TODO: for loop is very slow. find another way.
    dfi <- dataframe %>% filter(sentence_index == s_index)
    i1s <- filter(dfi, gold_edge==T)$i1
    i2s <- filter(dfi, gold_edge==T)$i2
    is <- sort(union(i1s, i2s))
    dfi <- mutate(
        dfi, word_i1 = match(i1, is),
        word_i2 = match(i2, is), 
        dep_len = word_i2 - word_i1
      ) %>%
      select(sentence_index, i1, i2, 
        dep_len, word_i1, word_i2, dep_len, everything())
    newdf <- rbind(newdf, dfi)
    setTxtProgressBar(pb, s_index / max(unique(dataframe$sentence_index)))
  }
  close(pb)
  return(newdf)
}

# saving/loading data (uncomment to save data)
data_dir <- "~/datadrive/pmi"

# # THIS TAKES AN HOUR (writes ~150MB file):
# # TODO: Should optimize this. But, it only need be run once.
# all_models <- lapply(all_models_raw, FUN = make_dep_len_nopunct)

# save(all_models, file = file.path(data_dir, "all_models.RData"))
# save(all_models, file = file.path(data_dir, "all_models_abs.RData"))
# save(all_models, file = file.path(data_dir, "wsj10cem_models.RData"))
# save(all_models, file = file.path(data_dir, "wsj10cem_models_abs.RData"))
load(file.path(data_dir, "all_models.RData"))     # to load all_models
# load(file.path(data_dir, "all_models_abs.RData")) # to load absolute-valued version

# # DO THE SAME FOR POS-CPMI RESULTS: takes a long time too
# all_pos_models <- lapply(all_pos_models_raw, FUN = make_dep_len_nopunct)

# save(all_pos_models, file = file.path(data_dir, "all_pos_models.RData"))
# save(all_pos_models, file = file.path(data_dir, "all_pos_models_abs.RData")) #or for absolute value
load(file.path(data_dir, "all_pos_models.RData"))     # to load all_pos_models
# load(file.path(data_dir, "all_pos_models_abs.RData"))     # to load all_pos_models_abs

dbert <- all_models[[1]]
bert <- all_models[[3]]
xlnet <- all_models[[4]]
xlm <- all_models[[6]]
bart <- all_models[[7]]
w2v <- all_models[[9]]
lstm <- all_models[[10]]
onlstm <- all_models[[11]]
onlstm_syd <- all_models[[12]]
baseline_linear <- all_models[[13]]
baseline_random <- all_models[[14]]

#### TABLE Overall accuracy scores total across all sentences ####

# Get total overall scores
total_uuas <- function(dataframe, pmi_edge_type = "pmi_edge_nonproj_sum") {
  n_edges <- length(which(dataframe$gold_edge == T))
  n_common <- length(which(dataframe$gold_edge == T & dataframe[pmi_edge_type] == T))
  uuas <- n_common / n_edges
  out_df <- uuas %>% as.data.frame(row.names = dataframe$model[[1]])
  colnames(out_df) <- "total_uuas"
  return(out_df)
}
binary_dist_total_precis <- function(dataframe, pmi_edge_type = "pmi_edge_nonproj_sum") {
  dataframe <- mutate(dataframe, longdep = dep_len > 1)
  #' Prepare csv as df data grouped by 'longdep'
  n_pmi_edges <- dataframe %>%
    filter(!!as.symbol(pmi_edge_type)==T) %>%
    group_by(longdep) %>%
    summarise(n=n())
  precis_df <- dataframe %>%
    filter(!!as.symbol(pmi_edge_type)==T) %>%
    mutate(common = gold_edge == !!as.symbol(pmi_edge_type)) %>%
    group_by(longdep, common) %>%
    summarise(n=n()) %>%
    pivot_wider(
      names_from = common, names_prefix = "n_pmi",
      values_from = c(n), values_fill = list(n = 0)
      ) %>%
    left_join(n_pmi_edges, by = c("longdep")) %>%
    mutate(precis = n_pmiTRUE / n)
  out_df = precis_df %>% 
    summarise(
      total_precis = mean(precis), n_pmiFALSE = sum(n_pmiFALSE),
      n_pmiTRUE = sum(n_pmiTRUE), n = sum(n),
      )
  return(out_df)
}
binary_dist_total_recall <- function(dataframe, pmi_edge_type = "pmi_edge_nonproj_sum") {
  dataframe <- mutate(dataframe, longdep = dep_len > 1)
  #' Prepare csv as df data grouped by 'longdep'
  n_gold_edges <- dataframe %>% 
    filter(relation!="NONE") %>%
    group_by(longdep) %>%
    summarise(n=n())
  recall_df <- dataframe %>% 
    filter(relation != "NONE") %>%
    mutate(common = gold_edge == !!as.symbol(pmi_edge_type)) %>%
    group_by(longdep, common) %>%
    summarise(n = n()) %>%
    pivot_wider(
      names_from = common, names_prefix = "n_pmi",
      values_from = c(n), values_fill = list(n = 0)
      ) %>%
    left_join(n_gold_edges, by = c("longdep")) %>%
    mutate(recall = n_pmiTRUE / n)
  out_df <- recall_df %>% 
    summarise(
      total_recall = mean(recall), n_pmiFALSE = sum(n_pmiFALSE),
      n_pmiTRUE = sum(n_pmiTRUE), n = sum(n),
      )
  return(out_df)
}
binary_dist_total_precis_recall <- function(dataframe, pmi_edge_type = "pmi_edge_nonproj_sum") {
  df <- left_join(
    binary_dist_total_precis(dataframe, pmi_edge_type),
    binary_dist_total_recall(dataframe, pmi_edge_type),
    by = c("longdep")
    )
  df %>% select(c("longdep", "total_precis", "total_recall")) %>%
    pivot_wider(
      names_from = longdep,
      values_from = c(total_precis, total_recall),
      names_prefix = "longdep",
      values_fill = list(n = 0)
      ) %>%
    add_column(model = dataframe$model[[1]])
}
get_total_uuas_table <- function(all_models, pmi_edge_type = "pmi_edge_nonproj_sum") {
  all_total_uuas_overall <- do.call(
    rbind, c(lapply(
      all_models,
      total_uuas,
      pmi_edge_type)))
  total_precis_recalls <- do.call(
    bind_rows, c(lapply(
      all_models,
      binary_dist_total_precis_recall,
      pmi_edge_type
    ))) %>%
    column_to_rownames("model")
  total_uuas_table <- cbind(all_total_uuas_overall, total_precis_recalls)
  return(total_uuas_table)
}

rounddf <- function(df, digits = 0, ignore_column = 1) {
  #' rounds entire dataframe (tibble),
  #' ignoring one column (default = 1st column), assumes rest are numeric.
  df[, -ignore_column] <- round(df[, -ignore_column], digits)
  return(df)
}

total_uuas_table <- get_total_uuas_table(all_models, pmi_edge_type = "pmi_edge_nonproj_sum")
total_uuas_table_proj <- get_total_uuas_table(all_models, pmi_edge_type = "pmi_edge_sum")

# To print to console for easy copying pasting to latex (ok, could make that smoother, but...)
# options(width = 1000)
# options(tibble.width = Inf)

total_uuas_tibble <- as_tibble(total_uuas_table[, c(1, 2, 4, 3, 5)], rownames = NA) %>% rownames_to_column(var="model")
total_uuas_tibble_proj <- as_tibble(total_uuas_table_proj[, c(1, 2, 4, 3, 5)], rownames = NA) %>% rownames_to_column(var="model")
print.data.frame(rounddf(total_uuas_tibble, 2), digits = 2)
print.data.frame(rounddf(total_uuas_tibble_proj, 2), digits = 2)

# write the table, with the columns in the right order
write.csv(total_uuas_table[, c(1, 2, 4, 3, 5)], "R/total_uuas_table.csv")
write.csv(total_uuas_table_proj[, c(1, 2, 4, 3, 5)], "R/total_uuas_table_proj.csv")

## POS CPMI version

total_pos_uuas_table <- get_total_uuas_table(all_pos_models, pmi_edge_type = "pmi_edge_nonproj_sum")
total_pos_uuas_table_proj <- get_total_uuas_table(all_pos_models, pmi_edge_type = "pmi_edge_sum")

total_pos_uuas_tibble <- as_tibble(total_pos_uuas_table[, c(1, 2, 4, 3, 5)], rownames = NA) %>% rownames_to_column(var="model")
total_pos_uuas_tibble_proj <- as_tibble(total_pos_uuas_table_proj[, c(1, 2, 4, 3, 5)], rownames = NA) %>% rownames_to_column(var="model")
print.data.frame(rounddf(total_pos_uuas_tibble, 2), digits = 2)
print.data.frame(rounddf(total_pos_uuas_tibble_proj, 2), digits = 2)

# write the table, with the columns in the right order
write.csv(total_pos_uuas_table[, c(1, 2, 4, 3, 5)], "R/total_pos_uuas_table.csv")
write.csv(total_pos_uuas_table_proj[, c(1, 2, 4, 3, 5)], "R/total_pos_uuas_table_proj.csv")


#### TABLE Overall accuracy scores avg by sentence ####

# Get avg overall scores
avg_uuas <- function(dataframe){
  n_edges = dataframe %>% filter(gold_edge==T) %>% group_by(sentence_index) %>% summarise(n=n()) # total number of edges
  acc_df = dataframe %>% filter(pmi_edge_sum==T,gold_edge==T)  %>% group_by(sentence_index) %>% summarise(n_acc=n()) %>%
    left_join(n_edges,by="sentence_index") %>% mutate(uuas=n_acc/n) %>% summarise(avg_uuas=mean(uuas))
  uuas = acc_df["avg_uuas"][[1]]
  out_df = uuas %>% as.data.frame(row.names=dataframe$model[[1]])
  colnames(out_df) = "avg_uuas"
  return(out_df)
}
all_avg_uuas_overall = do.call(rbind,c(lapply(all_models,avg_uuas)))
# Likewise for POS
pos_all_avg_uuas_overall = do.call(rbind,c(lapply(all_pos_models,avg_uuas)))

binary_dist_avg_precis <- function(dataframe){
  dataframe = mutate(dataframe, longdep=dep_len>1)
  #' Prepare csv as df data grouped by 'longdep'
  n_pmi_edges = dataframe %>% filter(pmi_edge_sum==T) %>% group_by(longdep,sentence_index) %>% summarise(n=n())
  precis_df = dataframe %>% filter(pmi_edge_sum==T) %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(longdep,acc,sentence_index) %>% summarise(n=n()) %>%
    pivot_wider(names_from = acc, names_prefix = "n_pmi", values_from = c(n), values_fill = list(n = 0)) %>%
    left_join(n_pmi_edges, by=c("longdep","sentence_index")) %>%
    mutate(precis = n_pmiTRUE/n)
  out_df = precis_df %>% summarise(avg_precis=mean(precis), n_pmiFALSE=sum(n_pmiFALSE),n_pmiTRUE=sum(n_pmiTRUE), n=sum(n))
  return(out_df)
}
binary_dist_avg_recall <- function(dataframe){
  dataframe = mutate(dataframe, longdep=dep_len>1)
  #' Prepare csv as df data grouped by 'longdep'
  n_gold_edges = dataframe %>% filter(relation!="NONE") %>% group_by(longdep,sentence_index) %>% summarise(n=n())
  recall_df = dataframe %>% filter(relation!="NONE") %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(longdep,acc,sentence_index) %>% summarise(n=n()) %>%
    pivot_wider(names_from = acc, names_prefix = "n_pmi", values_from = c(n), values_fill = list(n = 0)) %>%
    left_join(n_gold_edges, by=c("longdep","sentence_index")) %>%
    mutate(recall = n_pmiTRUE/n)
  out_df = recall_df %>% summarise(avg_recall=mean(recall), n_pmiFALSE=sum(n_pmiFALSE),n_pmiTRUE=sum(n_pmiTRUE), n=sum(n))
  return(out_df)
}
binary_dist_avg_precis_recall <- function(dataframe){
  df = left_join(binary_dist_avg_precis(dataframe), binary_dist_avg_recall(dataframe), by=c("longdep"))
  df %>% select(c("longdep","avg_precis","avg_recall")) %>%
    pivot_wider(names_from = longdep, values_from = c(avg_precis,avg_recall), names_prefix = "longdep", values_fill = list(n = 0)) %>%
    add_column(model = dataframe$model[[1]])
}
avg_precis_recalls = do.call(bind_rows,c(lapply(all_models, binary_dist_avg_precis_recall))) %>% column_to_rownames("model")
# Likewise for POS
pos_avg_precis_recalls = do.call(bind_rows,c(lapply(all_pos_models, binary_dist_avg_precis_recall))) %>% column_to_rownames("model")

# Combined table
avg_accuracy_table = cbind(all_avg_uuas_overall,avg_precis_recalls)
# write the table, with the columns in the right order
print(avg_accuracy_table[,c(1,2,4,3,5)], digits = 2)
# write.csv(avg_accuracy_table[,c(1,2,4,3,5)],"avg_accuracy_table.csv")

# Combined table for POS
pos_avg_accuracy_table = cbind(pos_all_avg_uuas_overall,pos_avg_precis_recalls)
# write the table, with the columns in the right order
print(pos_avg_accuracy_table[,c(1,2,4,3,5)], digits = 2)
# write.csv(pos_avg_accuracy_table[,c(1,2,4,3,5)],"avg_POS_accuracy_table.csv")



#### PLOTTING ####

#### Dep len histograms ####
prepare_df <- function(df){
  df <- df %>% mutate(acc=gold_edge==pmi_edge_sum)
  df$relation[is.na(df$relation)]<-"NONE"
  df <- df %>% prepare_POS() %>% add_class_predictor()
  return(df)
}


# quick histograms
gold.len <- bert %>% filter(gold_edge==T) %>% group_by(dep_len) %>% count
bert.len <- bert %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
xlnet.len <- xlnet %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
xlm.len <- xlm %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
# gpt2.len <- gpt2 %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
bart.len <- bart %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
dbert.len <- dbert %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
w2v.len <- w2v %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
baseline_rand.len <- baseline_random %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count

modelcols<-hue_pal()(6)

plothist<-function(df.len,fill,xlab,ylabel=T){
  yval = c(5, 10, 15, 20, 25)
  p<-df.len %>%  ggplot(aes(x=dep_len,y=n)) + geom_col(fill=fill,colour=fill) +
    scale_x_continuous(
      breaks = c(1,2,3,4,5,6,7,8,9),
      limits=c(0,10)
    ) +
    scale_y_continuous(labels = paste0(yval, "k"),
                       trans = "identity",
                       breaks = 10^3 * yval,
                       limits=c(0,26000)
    ) +
    theme(axis.title.y = element_blank()) +
    xlab(xlab)
  return(p)
}

hgold <- gold.len %>%
  ggplot(aes(x=dep_len,y=n)) + geom_col()  +
  scale_x_continuous(breaks = c(1,2,3,4,5,6,7,8,9), limits=c(0,10)) + xlab("gold")
hgold <- plothist(gold.len,"black", "gold")
hbart <- plothist(bart.len,modelcols[[1]], "Bart")
hbert <- plothist(bert.len,modelcols[[2]], "BERT")
hdbert <-plothist(dbert.len,modelcols[[3]],"DistilBERT")
hw2v <- plothist(w2v.len,modelcols[[4]],"Word2Vec")
hxlm <-  plothist(xlm.len,modelcols[[5]],  "XLM")
hxlnet <-plothist(xlnet.len,modelcols[[6]],"XLNet")
hrand <- plothist(baseline_rand.len,"darkgrey","random")
# hgpt2 <- plothist(gpt2.len,modelcols[[7]], "GPT2")

p.lindisthist <- grid.arrange(arrangeGrob(hgold,heights=unit(0.5,"npc")),
             arrangeGrob(hbart, hbert,  hdbert,
                         hw2v,  hxlnet, hxlm,
                         nrow=2),
             nrow=1,widths=c(1,3),
             top="Dependency arc length histograms")
ggsave("R/plots/lindisthist-norand.pdf",plot=p.lindisthist, width = 5, height = 3, units = "in")

p.lindisthist.rnd <- grid.arrange(arrangeGrob(hgold, hrand,nrow=2),
             arrangeGrob(hbart, hbert,  hdbert,
                         hw2v,  hxlnet, hxlm,
                         nrow=2),
             nrow=1,widths=c(1,3),
             top="Dependency arc length histograms")

ggsave("R/plots/lindisthist.pdf",plot=p.lindisthist.rnd, width = 5, height = 3, units = "in")

## Proportions length 1

# sum,      sum(abs)
gold.len[1,]$n/(bert %>% filter(gold_edge==T) %>% count())[[1]]
# .4896498, .4892236
bart.len[1,]$n/(xlm %>% filter(pmi_edge_sum==T) %>% count())[[1]]
# .6001901, .5779846
bert.len[1,]$n/(bert %>% filter(pmi_edge_sum==T) %>% count())[[1]]
# .7249265, .7519789
dbert.len[1,]$n/(dbert %>% filter(pmi_edge_sum==T) %>% count())[[1]]
# .6189005, .6536717
xlm.len[1,]$n/(xlm %>% filter(pmi_edge_sum==T) %>% count())[[1]]
# .5173889, .5220124
xlnet.len[1,]$n/(xlnet %>% filter(pmi_edge_sum==T) %>%  count())[[1]]
# .5547207, .5706027
# gpt2.len[1,]$n/(gpt2 %>% filter(pmi_edge_sum==T) %>%  count())[[1]]
w2v.len[1,]$n/(w2v %>% filter(pmi_edge_sum==T) %>%  count())[[1]]
# .4725134, .4718508
baseline_rand.len[1,]$n/(baseline_random %>% filter(pmi_edge_sum==T) %>%  count())[[1]]
# .3491135, .350123
#

#### Accuracy by dep len ####

prepare_by_len_gold <- function(dataframe){
  #' Prepare csv as df data grouped by 'dep_len'
  len = dataframe %>% filter(relation!="NONE") %>%
    group_by(dep_len) %>% summarise(meanpmi=mean(pmi_sum), varpmi=var(pmi_sum), n=n())
  dataframe = dataframe %>% filter(relation!="NONE") %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(dep_len,acc) %>% summarise(n=n()) %>%
    pivot_wider(names_from = acc, names_prefix = "pmi", values_from = c(n), values_fill = list(n = 0)) %>%
    left_join(len, by="dep_len") %>%
    mutate(pct_acc = pmiTRUE/n)
  return(dataframe)
}

xlnet.len.gold <- prepare_by_len_gold(xlnet)
bert.len.gold <-  prepare_by_len_gold(bert)
xlm.len.gold <-   prepare_by_len_gold(xlm)
bart.len.gold <-  prepare_by_len_gold(bart)
# gpt2.len.gold <-  prepare_by_len_gold(gpt2)
dbert.len.gold <- prepare_by_len_gold(dbert)
w2v.len.gold <-   prepare_by_len_gold(w2v)

lstm.len.gold <-   prepare_by_len_gold(lstm)
onlstm.len.gold <-   prepare_by_len_gold(onlstm)
onlstm_syd.len.gold <-   prepare_by_len_gold(onlstm_syd)

baseline_rand.len.gold <- prepare_by_len_gold(baseline_random)

join_five <- function(df1, df2, df3, df4, df5,
                      #' to full_join five data frames
                      by=c("n","meanlen"),
                      suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".XLM")){
  return(
    full_join(df1,df2,by=by,suffix=suffixes[1:2]) %>%
      full_join(rename_at(df3, vars(-by), function(x){paste0(x,suffixes[3])}), by=by) %>%
      full_join(rename_at(df4, vars(-by), function(x){paste0(x,suffixes[4])}), by=by) %>%
      full_join(rename_at(df5, vars(-by), function(x){paste0(x,suffixes[5])}), by=by) %>%
      pivot_longer(cols = -by, names_to = c(".value", "model"), names_pattern = "(.*)\\.(.*)"))
}

join_six <- function(df1, df2, df3, df4, df5, df6,
                      #' to full_join six data frames
                      by=c("n","meanlen"),
                      suffixes=c(".linear_baseline",".DistilBERT",".Bart",".BERT",".XLNet",".XLM")){
  return(
    full_join(df1,df2,by=by,suffix=suffixes[1:2]) %>%
      full_join(rename_at(df3, vars(-by), function(x){paste0(x,suffixes[3])}), by=by) %>%
      full_join(rename_at(df4, vars(-by), function(x){paste0(x,suffixes[4])}), by=by) %>%
      full_join(rename_at(df5, vars(-by), function(x){paste0(x,suffixes[5])}), by=by) %>%
      full_join(rename_at(df6, vars(-by), function(x){paste0(x,suffixes[6])}), by=by) %>%
      pivot_longer(cols = -by, names_to = c(".value", "model"), names_pattern = "(.*)\\.(.*)"))
}

# The five models in one df
five.len.gold <- join_five(dbert.len.gold,bart.len.gold,bert.len.gold,xlnet.len.gold,w2v.len.gold,
                           by = c("n","dep_len"),
                           suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".Word2Vec"))

five.len.gold.rnd <- join_five(dbert.len.gold,baseline_rand.len.gold,bert.len.gold,xlnet.len.gold,w2v.len.gold,
                               by = c("n","dep_len"),
                               suffixes=c(".DistilBERT",".random",".BERT",".XLNet",".Word2Vec"))
# reorder factor for plotting to make better sense
five.len.gold.rnd$model <- factor(five.len.gold.rnd$model,
                                  levels=c("BERT","DistilBERT","XLNet","Word2Vec","random"))


plotby_dep_len <- function(df, title="Accuracy (recall) by arc length",
                           recoloring=NULL, reshaping=NULL){
  p <- df %>% filter(n>60) %>%
    ggplot(aes(y=pct_acc, x=dep_len)) +
    geom_text(aes(label=n, y=Inf), hjust=0, size=2.5, colour="grey") +
    annotate("text",x=Inf,y=Inf, label="n", size=2.5, hjust=0, vjust=0, colour="grey") +
    geom_line(aes(group=dep_len), colour="grey") +
    geom_point(aes(size=n, colour=model, shape=model), fill="lightgrey", alpha=0.8) +
    coord_flip(clip = "off") +
    theme(legend.position="top", legend.box="vertical",legend.margin = margin(),
          plot.margin = ggplot2::margin(0, 50, 0, 2, "pt"),
          legend.spacing = unit(0, 'cm')
    ) + scale_x_continuous(trans="identity",breaks = seq(1, 23, by = 2), minor_breaks = seq(1, 23, by = 1)) +
    ylab("recall (# CPMI arc = gold arc)/(# gold arcs)") +
    xlab("arc length") +
    ggtitle(title)
  if(!is.null(recoloring)){
    p <- p + scale_color_manual(values=recoloring)
  }
  if(!is.null(reshaping)){
    p <- p + scale_shape_manual(values=reshaping)
  }
  return(p + guides(shape = guide_legend(override.aes = list(size = 3))))
}
p.dep_len <- plotby_dep_len(
  five.len.gold, title="Accuracy (recall) by arc length (n>60)")
ggsave("R/plots/acc_deplen.pdf",plot=p.dep_len, width = 5, height = 3.2, units = "in")

len.recolors <- hue_pal()(5)
len.recolors[[1]] <-modelcols[[2]]
len.recolors[[2]] <-modelcols[[3]]
len.recolors[[3]] <-modelcols[[6]]
len.recolors[[4]] <-modelcols[[4]]
len.recolors[[5]] <- "darkgrey"


len.shapes <- c(15,#bert
                18,#dbert
                19,#xlnet
                17,#w2v
                25#random
                )
p.dep_len.rnd <- plotby_dep_len(
  five.len.gold.rnd, title="Accuracy (recall) by arc length (n>60)",
  recoloring = len.recolors, reshaping = len.shapes)

ggsave("R/plots/acc_deplen-rand.pdf",plot=p.dep_len.rnd, width = 5, height = 3.2, units = "in")

#### Accuracy by gold dep label ####


prepare_by_relation <- function(dataframe,length_greater_than=0){
  #' Prepare csv as df data grouped by 'relation'
  relation_len <- dataframe %>% filter(gold_edge==T,
                                      dep_len>length_greater_than) %>%
    group_by(relation) %>% summarise(medlen=median(dep_len), meanlen=mean(dep_len), n=n(),
                                     meanpmi=mean(pmi_sum), varpmi=var(pmi_sum))
  dataframe <- dataframe %>% filter(gold_edge==T,
                                   dep_len>length_greater_than) %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(relation,acc) %>% summarise(n=n(), medlen=median(dep_len), meanlen=mean(dep_len)) %>%
    pivot_wider(names_from = acc, names_prefix = "pmi", values_from = c(n,medlen,meanlen), values_fill = list(n = 0)) %>%
    left_join(relation_len, by="relation") %>% mutate(pct_acc = n_pmiTRUE/n)
  return(dataframe)
}

xlnet.relation <-prepare_by_relation(xlnet)
bert.relation <- prepare_by_relation(bert)
xlm.relation <-  prepare_by_relation(xlm)
bart.relation <- prepare_by_relation(bart)
dbert.relation <-prepare_by_relation(dbert)
# gpt2.relation <- prepare_by_relation(gpt2)
w2v.relation  <- prepare_by_relation(w2v)

lstm.relation  <- prepare_by_relation(lstm)
onlstm.relation  <- prepare_by_relation(onlstm)
onlstm_syd.relation  <- prepare_by_relation(onlstm_syd)

baseline_rand.relation <- prepare_by_relation(baseline_random)
baseline_linear.relation <- prepare_by_relation(baseline_linear)

five.relation <- join_five(
  dbert.relation,bart.relation,bert.relation,xlnet.relation,w2v.relation,
  by=c("n","relation","meanlen"),
  suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".Word2Vec"))

five.relation.rnd <- join_five(
  dbert.relation,baseline_rand.relation,bert.relation,xlnet.relation,w2v.relation,
  by=c("n","relation","meanlen"),
  suffixes=c(".DistilBERT",".random",".BERT",".XLNet",".Word2Vec"))
five.relation.rnd$model <- factor(
  five.relation.rnd$model,
  levels=c("BERT","DistilBERT","XLNet","Word2Vec","random"))

six.relation <- join_six(
  baseline_linear.relation,dbert.relation,baseline_rand.relation,bert.relation,xlnet.relation,w2v.relation,
  by=c("n","relation","meanlen"),
  suffixes=c(".connect-adjacent",".DistilBERT",".random",".BERT",".XLNet",".Word2Vec"))
six.relation$model <- factor(
  six.relation$model,
  levels=c("BERT","DistilBERT","XLNet","Word2Vec","random","connect-adjacent"))

## same, only for arc-length ≥ 1

xlnet.relation.gt1<- prepare_by_relation(xlnet,length_greater_than = 1)
bert.relation.gt1 <- prepare_by_relation(bert,length_greater_than = 1)
xlm.relation.gt1 <-  prepare_by_relation(xlm,length_greater_than = 1)
bart.relation.gt1 <- prepare_by_relation(bart,length_greater_than = 1)
dbert.relation.gt1 <-prepare_by_relation(dbert,length_greater_than = 1)
# gpt2.relation.gt1 <- prepare_by_relation(gpt2,length_greater_than = 1)
w2v.relation.gt1 <-  prepare_by_relation(w2v, length_greater_than = 1)

lstm.relation.gt1        <- prepare_by_relation(lstm, length_greater_than = 1)
onlstm.relation.gt1      <- prepare_by_relation(onlstm, length_greater_than = 1)
onlstm_syd.relation.gt1  <- prepare_by_relation(onlstm_syd, length_greater_than = 1)

baseline_rand.relation.gt1 <- prepare_by_relation(baseline_random,length_greater_than = 1)
baseline_linear.relation.gt1 <- prepare_by_relation(baseline_linear,length_greater_than = 1)

five.relation.gt1 <- join_five(dbert.relation.gt1,bart.relation.gt1,bert.relation.gt1,xlnet.relation.gt1,w2v.relation.gt1,
                               by=c("n","meanlen"),
                               suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".Word2Vec"))

five.relation.gt1.rnd <- join_five(dbert.relation.gt1,baseline_rand.relation.gt1,bert.relation.gt1,xlnet.relation.gt1,w2v.relation.gt1,
                               by=c("n","meanlen"),
                               suffixes=c(".DistilBERT",".random",".BERT",".XLNet",".Word2Vec"))
five.relation.gt1.rnd$model <- factor(five.relation.gt1.rnd$model,
                                  levels=c("BERT","DistilBERT","XLNet","Word2Vec","random"))

# A plot exploring accuracy by relation with respect to linear distance, model, and n
plotby_rel <- function(df, title="all arc lengths",ylabel=T,recoloring=NULL,reshaping=NULL) {
  p <- df %>%  filter(n>60) %>%
    ggplot(aes(y=pct_acc, x=reorder(relation, meanlen))) +
    annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0,colour="grey") +
    geom_text(aes(label=paste("",n,sep=""),y=Inf), hjust=0, size=3, colour="grey") +  # to print n
    annotate("text",x=Inf,y=-Inf, label="mean arclength", size=3, hjust=0, vjust=0) +
    geom_text(aes(label=round(meanlen, digits=1), y=-Inf), hjust=0, size=3, alpha=0.2) +
    geom_line(aes(group=relation), colour="grey") +
    geom_point(aes(size=n, colour=model, shape=model), fill="lightgrey", alpha=0.6) +
    coord_flip(clip = "off") +
    theme(legend.position="top", legend.box="vertical",legend.margin = margin(),
          plot.margin = ggplot2::margin(0, 50, 2, 2, "pt"),
          axis.ticks = element_blank()) +
    ylab(NULL) +
    ggtitle(title) +
    theme(plot.title = element_text(hjust = 0.5),
          legend.spacing = unit(0, 'cm')) 
  if (ylabel) {p <- p + xlab("gold dependency label (ordered by mean arc length)")}
  else p <- p + xlab(NULL)
  if(!is.null(recoloring)){
    p <- p + scale_color_manual(values=recoloring)
  }
  if(!is.null(reshaping)){
    p <- p + scale_shape_manual(values=reshaping)
  }
  return(p + guides(shape = guide_legend(override.aes = list(size = 3))))
}

#extract legend
#https://github.com/hadley/ggplot2/wiki/Share-a-legend-between-two-ggplot2-graphs
g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}

hbart <- plothist(bart.len,modelcols[[1]], "Bart")
hbert <- plothist(bert.len,modelcols[[2]], "BERT")
hdbert <-plothist(dbert.len,modelcols[[3]],"DistilBERT")
hw2v <- plothist(w2v.len,modelcols[[4]],"Word2Vec")
hxlm <-  plothist(xlm.len,modelcols[[5]],  "XLM")
hxlnet <-plothist(xlnet.len,modelcols[[6]],"XLNet")

rel.recolors <- hue_pal()(5)
rel.recolors[[1]] <-modelcols[[1]]
rel.recolors[[2]] <-modelcols[[2]]
rel.recolors[[3]] <-modelcols[[3]]
rel.recolors[[4]] <-modelcols[[4]]
rel.recolors[[5]] <-modelcols[[6]]


p.rel <- plotby_rel(five.relation, title="all arc lengths",ylabel=T,recoloring = rel.recolors)
p.rel.gt1<- plotby_rel(five.relation.gt1, title="arc length > 1",ylabel=F,recoloring = rel.recolors)
p.rel.both <- grid.arrange(
  g_legend(p.rel.gt1),
  arrangeGrob(
    p.rel + theme(legend.position="none"),
    p.rel.gt1 + theme(legend.position="none"),
    nrow=1),
  nrow=2,heights=c(15, 100),
  top="Accuracy (recall) by gold label (only labels with n>60)",
  bottom="recall (# CPMI arc = gold arc)/(# gold arcs)")
ggsave("R/plots/acc_label.pdf",plot=p.rel.both, width = 9.7, height = 4.75, units = "in")


rel.recolors.rnd <- len.recolors
rel.shapes <- len.shapes

# p.rel.rnd <- plotby_rel(
#   five.relation.rnd, title="all arc lengths",
#   recoloring = rel.recolors.rnd, reshaping = rel.shapes)
# p.rel.gt1.rnd<- plotby_rel(
#   five.relation.gt1.rnd, title="arc length > 1",ylabel=F,
#   recoloring = rel.recolors.rnd, reshaping = rel.shapes)
# p.rel.both.rnd <- grid.arrange(
#   g_legend(p.rel.gt1.rnd),
#   arrangeGrob(
#     p.rel.rnd + theme(legend.position="none"),
#     p.rel.gt1.rnd + theme(legend.position="none"),
#     nrow=1),
#   nrow=2,heights=c(15, 100),
#   top="Accuracy (recall) by gold label (only labels with n>60)",
#   bottom="recall (# CPMI arc = gold arc)/(# gold arcs)")
# ggsave("R/plots/acc_label-rand.pdf",plot=p.rel.both.rnd, width = 9.7, height = 4.75, units = "in")

rel.shapes_with_linear <- len.shapes
rel.shapes_with_linear[6] <- 4
rel.recolors_with_linear <- rel.recolors.rnd
rel.recolors_with_linear[6] <- "black"

p.rel.rndlin <- plotby_rel(
    six.relation, title="all arc lengths",
    recoloring = rel.recolors_with_linear, reshaping = rel.shapes_with_linear) +
  ylim(-0.025,1.0) + 
  guides(color = guide_legend(nrow=1), shape = guide_legend(nrow = 1,override.aes = list(size = 3)))
p.rel.gt1.rnd<- plotby_rel(
  five.relation.gt1.rnd, title="arc length > 1",ylabel=F,
  recoloring = rel.recolors.rnd, reshaping = rel.shapes)+
  ylim(-0.0125,0.5)
p.rel.both.rndlin <- grid.arrange(
  g_legend(p.rel.rndlin),
  arrangeGrob(
    p.rel.rndlin + theme(legend.position="none"),
    p.rel.gt1.rnd + theme(legend.position="none"),
    nrow=1),
  nrow=2,heights=c(15, 100),
  top="Accuracy (recall) by gold label (only labels with n>60)",
  bottom="recall (# CPMI arc = gold arc)/(# gold arcs)")
ggsave("R/plots/acc_label_with_lin.pdf",plot=p.rel.both.rndlin, width = 9.7, height = 4.75, units = "in")

#### Relation Table ####


# dbert <- all_models[[1]]
# bert <- all_models[[3]]
# xlnet <- all_models[[4]]
# xlm <- all_models[[6]]
# bart <- all_models[[7]]
# w2v <- all_models[[9]]
# lstm <- all_models[[10]]
# onlstm <- all_models[[11]]
# onlstm_syd <- all_models[[12]]
# baseline_linear <- all_models[[13]]
# baseline_random <- all_models[[14]]

join_11 <- function(
    df1, df2, df3, df4, df5, df6,
    df7, df8, df9, df10, df11,
    suffixes,
    by=c("n","meanlen")){
  #' to full_join 11 data frames
  newdf <- full_join(df1, df2, by=by, suffix=suffixes[1:2]) %>%
      full_join(rename_at(df3, vars(-by), function(x){paste0(x,suffixes[3])}), by=by) %>%
      full_join(rename_at(df4, vars(-by), function(x){paste0(x,suffixes[4])}), by=by) %>%
      full_join(rename_at(df5, vars(-by), function(x){paste0(x,suffixes[5])}), by=by) %>%
      full_join(rename_at(df6, vars(-by), function(x){paste0(x,suffixes[6])}), by=by) %>%
      full_join(rename_at(df7, vars(-by), function(x){paste0(x,suffixes[7])}), by=by) %>%
      full_join(rename_at(df8, vars(-by), function(x){paste0(x,suffixes[8])}), by=by) %>%
      full_join(rename_at(df9, vars(-by), function(x){paste0(x,suffixes[9])}), by=by) %>%
      full_join(rename_at(df10,vars(-by), function(x){paste0(x,suffixes[10])}), by=by) %>%
      full_join(rename_at(df11,vars(-by), function(x){paste0(x,suffixes[11])}), by=by) %>%
      pivot_longer(cols = -by, names_to = c(".value", "model"), names_pattern = "(.*)\\.(.*)")
  return(newdf)
}

rel.all <- join_11(
  bert.relation,
  dbert.relation,
  bart.relation,
  xlnet.relation,
  xlm.relation,
  w2v.relation,
  lstm.relation,
  onlstm.relation,
  onlstm_syd.relation,
  baseline_linear.relation,
  baseline_rand.relation,
  c(
    ".BERT",
    ".DistilBERT",
    ".Bart",
    ".XLNet",
    ".XLM",
    ".W2V",
    ".LSTM",
    ".ONLSTM",
    ".ONLSTM-SYD",
    ".connect-adjacent",
    ".random"
    ),
  by=c("n","relation","meanlen")
  ) 
  
relation_comparison <- rel.all %>%
  select(relation, model, meanlen, pct_acc, n) %>%
  pivot_wider(names_from="model", values_from="pct_acc")

relation_comparison$`XLNet/adj`<- relation_comparison$XLNet / relation_comparison$`connect-adjacent`
relation_comparison$`XLNet-adj`<- relation_comparison$XLNet - relation_comparison$`connect-adjacent`
relation_comparison$better_baseline <- pmax(relation_comparison$`connect-adjacent`, relation_comparison$random)
relation_comparison$`XLNet/baselines`<- relation_comparison$XLNet / relation_comparison$better_baseline
relation_comparison$`XLNet-baselines`<- relation_comparison$XLNet - relation_comparison$better_baseline
relation_tibble <- arrange(
      relation_comparison,
      desc(`XLNet/baselines`)) 

print.data.frame(
  rounddf(
    relation_tibble %>% filter(n>60) %>% select(-c("better_baseline", "XLNet/adj", "XLNet-adj", "XLNet-baselines")), 
    2),
  digits = 2)


p.rel.all <- plotby_rel(
    rel.all, title="all arc lengths",
    reshaping=seq(11)) 
ggsave("R/plots/TEST.pdf",plot=p.rel.all, width = 8.75, height = 6.75, units = "in")


#### Jaccard for model comparison ####

jaccard_simil_cols <- function(d) {
  proxy::simil(t(d %>% select(-c("sentence_index","i1","i2"))), method = 'jaccard')
}
plotjacc <- function(M){
  col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
  corrplot::corrplot(M, method="color", col=col(200),
                     cl.pos = "n",#tl.pos = "n",
                     type="upper", order="FPC",
                     addCoef.col = "black", # Add coefficient of correlation
                     tl.col="black", tl.srt=35, #Text label color and rotation
                     diag=F,
                     mar = c(0, 0, 0, 0))
}

all_models[[16]] <- all_models[[14]] %>% rename(pmi_edge_sum_discarded = pmi_edge_sum, pmi_edge_sum = pmi_edge_nonproj_sum)
all_models[[16]]$model[all_models[[16]]$model=="baseline_random"]<-"random nonproj."
# remove the pmi_sum column because it's only defined for the models, and we don't need it
all_no_pmi_sum <- bind_rows(lapply(
  all_models, 
  FUN = function(x) {x %>% select(-c(pmi_sum))} )) %>%
  filter(!model %in% c("ONLSTM-SYD","ONLSTM","LSTM","GPT2","baseline_random2"))

compare <- all_no_pmi_sum %>%
  group_by(sentence_index, i1, i2, model) %>% 
  summarise(edge = pmi_edge_sum) %>%
  pivot_wider(names_from = model, values_from = edge) %>% 
  ungroup() %>% 
  rename("random proj." = baseline_random, "connect adj." = baseline_linear)

jacc <- jaccard_simil_cols(compare)
pdf("R/plots/compare_jacc_sum.pdf", width = 6, height = 6)
plotjacc(as.matrix(jacc))
dev.off()


## Getting ppl plot ####

results_dir <- "results/results-clean"
scores_csv_file_loc <- "scores*.csv" # for non-absolute value

if (USE_WSJ10) {results_dir <- "results/results-comparison-wsj10"}
if (USE_ABSOLUTE_VALUE) {scores_csv_file_loc <- "loaded*/wordpair*.csv"}

prepareppl <- function(csv){
  #' Prepare the raw data
  df = csv %>%
    mutate(sentence_logperplexity=-pseudo_loglik/sentence_length) %>%
    select(sentence_index, sentence_length, number_edges, pseudo_loglik, sentence_logperplexity,
           baseline_linear, baseline_random_proj, baseline_random_nonproj,
           # too many types of scores makes it messy. just choosing the 'projective' types
           projective.uuas.sum, projective.uuas.tril, projective.uuas.triu)  %>%
    pivot_longer(
      cols = -c(sentence_index, sentence_length, number_edges, pseudo_loglik, sentence_logperplexity),
      names_to = "score_method", values_to = "uuas")
  return(df)
}

scores_bart <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "cems/bart-large*", scores_csv_file_loc))))
scores_bert_base <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "cems/bert-base-cased*", scores_csv_file_loc))))
scores_bert_large <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "cems/bert-large-cased*", scores_csv_file_loc))))
scores_dbert <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "cems/distilbert-base-cased*", scores_csv_file_loc))))
scores_gpt2 <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "cems/gpt2*", scores_csv_file_loc))))
scores_w2v <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "cems/w2v*", scores_csv_file_loc))))
scores_xlm <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "cems/xlm-mlm-en-2048*", scores_csv_file_loc))))
scores_xlnet_base <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "cems/xlnet-base-cased*", scores_csv_file_loc))))
scores_xlnet_large <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "cems/xlnet-large-cased*", scores_csv_file_loc))))

scores_lstm <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "lstms/loaded=lstm_pad0*", scores_csv_file_loc))))
scores_onlstm <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "lstms/loaded=onlstm_pad0*", scores_csv_file_loc))))
scores_onlstm_syd <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "lstms/loaded=onlstm_syd_pad0*", scores_csv_file_loc))))

scores_baseline_linear <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "baselines/linear_baseline*", "scores*.csv"))))
scores_baseline_random <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "baselines/random_baseline_pad0_1_*", "scores*.csv"))))
scores_baseline_random2 <- prepareppl(read_csv(Sys.glob(file.path(
  results_dir, "baselines/random_baseline_pad0_2_*", "scores*.csv"))))

scores_bart$model  <- "Bart"
scores_bert_base$model  <- "BERT-base"
scores_bert_large$model  <- "BERT-large"
scores_dbert$model <- "DistilBERT"
scores_gpt2$model  <- "GPT2"
scores_w2v$model   <- "Word2Vec"
scores_xlm$model   <- "XLM"
scores_xlnet_large$model <- "XLNet-large"
scores_xlnet_base$model <- "XLNet-base"

# scores_lstm$model <- "LSTM"
# scores_onlstm$model <- "ONLSTM"
# scores_onlstm_syd$model <- "ONLSTM-SYD"

# scores_baseline_linear$model <- "baseline_linear"
# scores_baseline_random$model <- "baseline_random"
# scores_baseline_random2$model <- "baseline_random2"

scores_all_models <- rbind(
  scores_dbert,
  scores_bert_base,
  scores_bert_large,
  scores_xlnet_base,
  scores_xlnet_large,
  scores_xlm,
  scores_bart)
  # scores_gpt2,
  # scores_w2v,
  # scores_lstm,
  # scores_onlstm,
  # scores_onlstm_syd,
  # scores_baseline_linear,
  # scores_baseline_random,
  # scores_baseline_random2



make_uuas_vs_ppl_plot <- function(df) {
  plot <- df %>% 
    filter(
      score_method %in% c("projective.uuas.sum"),
      sentence_length > 4,
      sentence_logperplexity<6*exp(1)) %>%
    ggplot(aes(x=sentence_logperplexity,y=uuas)) + 
    theme(legend.position = "top") +
    geom_point(alpha=.15) + 
    geom_smooth(method = "lm", formula = y~x) +
    # geom_boxplot(aes(x=0),alpha=0.15) +
    scale_y_continuous(breaks=c(0,.5,1)) +
    facet_grid(model~.) +
    ggpmisc::stat_poly_eq(aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~~")),
                label.x = "right", label.y = 1,
                formula = y~x, parse = TRUE, size = 3) +
    labs(x="log(pseudo-perplexity)", y="UUAS") +
    ggtitle("Accuracy vs LM performance")
}
p.uuas_vs_ppl<-make_uuas_vs_ppl_plot(scores_all_models)

ggsave("R/plots/uuas_vs_ppl.pdf",plot=p.uuas_vs_ppl, width = 4.5, height = 6, units = "in")
p.uuas_vs_ppl_small<-make_uuas_vs_ppl_plot(
  scores_all_models %>% filter(model %in% c("BERT-base", "XLNet-base")))
ggsave("R/plots/uuas_vs_ppl_small.pdf",plot=p.uuas_vs_ppl_small, width = 4.5, height = 2.2, units = "in")
