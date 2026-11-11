library(tidyverse)
library(data.table)
library(stringr)
library(magrittr)
library(ggplot2)
library(ggpubr)
library(ggthemes)
library(cowplot)
library(rstatix)
library(lme4)
library(lmerTest)
source(file.path("C:\\Users\\cstone\\OneDrive - UNSW\\Documents\\Projects\\honours_projects\\Honours2026\\code\\analysis\\my_theme.R"))

## Set paths -------------------------------------------------------------------
path_exp <- "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/honours_projects/Honours2026/"
path_data <- paste0(path_exp, 'data/derivatives/processed')
 
## Import data -----------------------------------------------------------------
d_rat <- fread(file.path(path_data, 'Honours2026_eeg_coherence_State_tar-long.csv'))
d_conf <- fread(file.path(path_data, 'Honours2026_eeg_coherence_Confidence_cong-tar-long.csv'))

## Clean data ------------------------------------------------------------------
exclude <- c("sub-05", "sub-10", "sub-21", "sub-24", "sub-40") # 'sub-37

# clean ratio data
d_rat <- d_rat[!(Subject %in% exclude)]
select_cols <- c(paste0('T', 1434:5939), 'Subject', 'Condition') # -0.1 - 2.1 s post-PLH period (i.e., 0.7*2048 = 1024, 2.9*2048= 5934)
d_rat <- d_rat[, ..select_cols] 

# clean confidence data
d_conf <- d_conf[!(Subject %in% exclude)]
select_cols <- c(paste0('T', 1434:5939), 'Subject', 'Condition') # -0.1 - 2.1 s post-PLH period (i.e., 0.7*2048 = 1024, 2.9*2048= 5934)
d_conf <- d_conf[, ..select_cols]

## Run analysis ----------------------------------------------------------------
# write function to extract p_vals
extract_t_1samp <- function(x) {
  res <- t.test(x, mu=0)
  p_val <- res$p.value
  return(p_val)
}

n_tpoints_tar_long = 4506
ts_tar_long <- seq(-0.1, 2.1, length.out=n_tpoints_tar_long)
select_cols_tar_long <- c(paste0('T', 1434:5939))
# Ratio ------------------------------------------------------------------------
# CHECK OUTLIERS
x<-d_rat[Condition=='LOW', 
         melt(.SD, 
              id.vars=c('Subject', 'Condition'),
              variable.name='tpoint',
              value.name='Coherence'),
         .SDcols=c(select_cols_tar_long, 'Subject', 'Condition')]

ggplot() + geom_line(data=x, aes(tpoint, Coherence, color=Subject, group=Subject))

# HIGH
# x[tpoint == 'T4503', .(z=scale(Coherence))][order(z.V1)] # sub-37
# x[tpoint == 'T5239', .(z=scale(Coherence))][order(z.V1)] # sub-01 # post-target
# # MEDHIGH
# x[tpoint == 'T2917', .(z=scale(Coherence))][order(z.V1)] # sub-37
# x[tpoint == 'T2960', .(z=scale(Coherence))][order(z.V1)] # sub-37 # post-target
# # MEDLOW
# x[tpoint == 'T4035', .(z=scale(Coherence))][order(z.V1)] # sub-37
# x[tpoint == 'T1110', .(z=scale(Coherence))][order(z.V1)] # sub-37 # post-target
# x[tpoint == 'T3420', .(z=scale(Coherence))][order(z.V1)] # sub-04
# # LOW
# x[tpoint == 'T4307', .(z=scale(Coherence))][order(z.V1)] # sub-12
# x[tpoint == 'T1444', .(z=scale(Coherence))][order(z.V1)] # sub-12 # post-target
exclude <- c('sub-37')  #'sub-01', 'sub-04', 'sub-12',


## Perform 1-sample t-test across conditions
rslts_ratav_pval <- d_rat[!(Subject %in% exclude),
                          lapply(.SD, mean), .SDcols=select_cols_tar_long, by=Subject
][, lapply(.SD, extract_t_1samp), .SDcols=select_cols_tar_long
][, melt(.SD, 
         variable.name='tpoint',
         value.name='pval'),
  .SDcols=select_cols_tar_long]

rslts_ratav_pval <- rslts_ratav_pval[, pval_adj := p.adjust(pval, method = 'fdr')]
rslts_ratav_pval[, siglines := case_when(pval_adj < .05 ~ -0.01)]
sig_tpoints <- rslts_ratav_pval[!(is.na(siglines)), tpoint]
rslts_ratav_pval$time <- rep(ts_tar_long, times=1)

## Perform 1-sample t-tests for each condition
rslts_rat_pval <- d_rat[!(Subject %in% exclude),
                          lapply(.SD, extract_t_1samp), 
                          .SDcols=select_cols_tar_long,
                          by=c('Condition')
][, 
  melt(.SD, id.vars='Condition',
       variable.name='tpoint',
       value.name='pval'),
  .SDcols=c(select_cols_tar_long, 'Condition'),
][order(Condition, tpoint)]

rslts_rat_pval <- rslts_rat_pval[, pval_adj := p.adjust(pval, method = 'fdr'), 
                                  by=Condition]
rslts_rat_pval[, siglines := case_when(Condition == 'HIGH' & pval_adj < .05 ~ -0.012,
                                       Condition == 'MEDHIGH' & pval_adj < .05 ~ -0.014,
                                       Condition == 'MEDLOW' & pval_adj < .05 ~ -0.016,
                                       Condition == 'LOW' & pval_adj < .05 ~ -0.018,
                                          .default =  NA)]
rslts_rat_pval$time <- rep(ts_tar_long, times=4)

## average over data for plotting
#select_cols <- c(paste0('T', 1434:4301))
d_rat_p <- d_rat[!(Subject %in% exclude), 
                 lapply(.SD, mean), 
                 by=Condition, 
                 .SDcols=select_cols_tar_long
][, melt(.SD, 
         id.vars='Condition',
         variable.name='tpoint',
         value.name='Coherence'),
  .SDcols=c(select_cols_tar_long, 'Condition')
][order(Condition, tpoint)]


d_rat_se <- d_rat[!(Subject %in% exclude), 
                 lapply(.SD, sd), 
                 by=Condition, 
                 .SDcols=select_cols_tar_long
][, melt(.SD, 
         id.vars='Condition',
         variable.name='tpoint',
         value.name='SD_Coherence'),
  .SDcols=c(select_cols_tar_long, 'Condition')
][order(Condition, tpoint)]
d_rat_p$SE_Coherence <- d_rat_se[, SD_Coherence/sqrt(34)]

# add column for time in ms
d_rat_p$time <- rep(ts_tar_long, times=4)

# create plot 
plot_rat <- ggplot() + 
  
  # add reference lines
  geom_line(data=data.frame(x=c(0, 0), y=c(-0.01, 0.01)), 
            aes(x, y), linetype = 3, linewidth=0.8, alpha = 0.4) +
  geom_line(data=data.frame(x=c(1.3, 1.3), y=c(-0.01, 0.01)), 
            aes(x, y), linetype = 3, linewidth=0.8, alpha = 0.4) +
  geom_line(data=data.frame(x=c(-0.1, 2), y=c(0, 0)), 
            aes(x, y), linetype = 3, linewidth=0.8, alpha = 0.4) +
  
  # add data
  geom_line(data=d_rat_p, 
            aes(x=time, 
                y=Coherence, 
                color=Condition),
            linewidth=1) +
  
  geom_ribbon(data=d_rat_p, 
              aes(x=time, y=Coherence, 
                  ymin=Coherence-SE_Coherence, ymax=Coherence+SE_Coherence,
                  fill=Condition), 
              linetype=0, colour=NA, alpha=0.1, show.legend = F) +
  
  # add significance lines for t-tests
  geom_line(data=rslts_rat_pval,
            aes(x=time,
                y=siglines,
                color=Condition),
            linewidth=1) +
  geom_line(data=rslts_ratav_pval,
            aes(x=time,
                y=siglines),
            linewidth=1) +

  # customise
  scale_y_continuous(name='Coherence diff. (HPL - LPL)',
                     breaks=c(-0.02, -0.01, 0, 0.01, 0.02),
                     labels=c('-0.02', '-0.01', '0.00', '0.01', '0.02'),
                     limits=c(-0.035, 0.035),
                     expand=expansion(mult = 0.1)) +
  # scale_x_continuous(name='Time from placeholder onset (s)',
  #                    breaks=c(0, 0.4, 0.8, 1.2),
  #                    labels=c('0.0', '0.4', '0.8', '1.2')) +
  scale_x_continuous(name='Time from placeholder onset (s)',
                     breaks=c(0, 0.5, 1.0, 1.5, 2.0),
                     labels=c('0.0', '0.5', '1.0', '1.5', '2.0')) +
  scale_color_manual(name='Condition',
                     breaks=c('HIGH', 'MEDHIGH', 'MEDLOW', 'LOW'),
                     values=c('HIGH' = '#9467bd',
                              'MEDHIGH' = '#17becf',
                              'MEDLOW' = '#e377c2',
                              'LOW' = '#ff7f0e'),
                     labels=c('0.9', '0.8', '0.7', '0.6')) +
  # geom_rangeframe(data=data.frame(x=c(0, 01.2), y=c(-0.02, 0.03)),
  #                 aes(x, y), size=1, color='black') +
  geom_rangeframe(data=data.frame(x=c(0, 2), y=c(-0.02, 0.02)),
                  aes(x, y), size=1, color='black') +
  my_theme() + theme(legend.position = 'top')


svg("C:\\Users\\cstone\\OneDrive - UNSW\\Documents\\Projects\\honours_projects\\Honours2026\\output\\fig_coherence_rat_N=34_tar.svg",
    width=9, height=4)
plot(plot_rat)
dev.off()

# Confidence  ------------------------------------------------------------------
# CHECK OUTLIERS
x<-d_conf[Condition=='HIGH', 
         melt(.SD, 
              id.vars=c('Subject', 'Condition'),
              variable.name='tpoint',
              value.name='Coherence'),
         .SDcols=c(select_cols_tar_long, 'Subject', 'Condition')]

ggplot() + geom_line(data=x, aes(tpoint, Coherence, color=Subject, group=Subject))

# LOW
x[tpoint == 'T3903', .(z=scale(Coherence))][order(z.V1)] # sub-37
# HIGH
x[tpoint == 'T3401', .(z=scale(Coherence))][order(z.V1)] # sub-03
x[tpoint == 'T5237', .(z=scale(Coherence))][order(z.V1)] # sub-01

exclude <- c('sub-37') # 'sub-01, 'sub-03

## Perform 1-sample t-test across conditions
rslts_confav_pval <- d_conf[!(Subject %in% exclude),
                          lapply(.SD, mean), .SDcols=select_cols_tar_long, by=Subject
][, lapply(.SD, extract_t_1samp), .SDcols=select_cols_tar_long,
][, melt(.SD, 
         variable.name='tpoint',
         value.name='pval'),
  .SDcols=select_cols_tar_long]

rslts_confav_pval <- rslts_confav_pval[, pval_adj := p.adjust(pval, method = 'fdr')]
rslts_confav_pval[, siglines := case_when(pval_adj < .05 ~ -0.01)]
sig_tpoints <- rslts_confav_pval[!(is.na(siglines)), tpoint]
rslts_confav_pval$time <- rep(ts_tar_long, times=1)

## Perform 1-sample t-tests for each condition
rslts_conf_pval <- d_conf[!(Subject %in% exclude),
                        lapply(.SD, extract_t_1samp), 
                        .SDcols=select_cols_tar_long,
                        by=c('Condition')
][, 
  melt(.SD, id.vars='Condition',
       variable.name='tpoint',
       value.name='pval'),
  .SDcols=c(select_cols_tar_long, 'Condition'),
][order(Condition, tpoint)]

rslts_conf_pval <- rslts_conf_pval[, pval_adj := p.adjust(pval, method = 'fdr'), 
                                 by=Condition]
rslts_conf_pval[, siglines := case_when(Condition == 'HIGH' & pval_adj < .05 ~ -0.012,
                                       Condition == 'LOW' & pval_adj < .05 ~ -0.014,
                                       .default =  NA)]
rslts_conf_pval$time <- rep(ts_tar_long, times=2)

## Perform paired t-test between conditions
rslts_conf_paired <- data.table()

for (t in select_cols_tar_long) {
  
  # run test for each effect
  t_conf <- t.test(x=d_conf[!(Subject %in% exclude) & Condition == 'HIGH', get(t)],
                   y=d_conf[!(Subject %in% exclude) & Condition == 'LOW', get(t)],
                   mu=0, paired=T)

  t_df <- data.table(tpoint = t,
                     p = t_conf$p.value)
  
  rslts_conf_paired <- rbind(rslts_conf_paired, t_df)
}

rslts_conf_paired <- rslts_conf_paired[, pval_adj := p.adjust(p, method = 'fdr')]
rslts_conf_paired[, siglines := case_when(pval_adj < .05 ~ -0.016, .default =  NA)]
rslts_conf_paired$time <- rep(ts_tar_long, times=1)

# average over data for plotting
d_conf_p <- d_conf[!(Subject %in% exclude), 
                 lapply(.SD, mean), 
                 by=Condition, 
                 .SDcols=select_cols_tar_long
][, melt(.SD, 
         id.vars='Condition',
         variable.name='tpoint',
         value.name='Coherence'),
  .SDcols=c(select_cols_tar_long, 'Condition'),
][order(Condition, tpoint)]


d_conf_se <- d_conf[!(Subject %in% exclude), 
                  lapply(.SD, sd), 
                  by=Condition, 
                  .SDcols=select_cols_tar_long
][, melt(.SD, 
         id.vars='Condition',
         variable.name='tpoint',
         value.name='SD_Coherence'),
  .SDcols=c(select_cols_tar_long, 'Condition')
][order(Condition, tpoint)]
d_conf_p$SE_Coherence <- d_conf_se[, SD_Coherence/sqrt(34)]


# add column for time in ms
d_conf_p$time <- rep(ts_tar_long, times=2)

# create plot 
plot_conf <- ggplot() + 
  
  # add reference lines
  geom_line(data=data.frame(x=c(0, 0), y=c(-0.01, 0.01)), 
            aes(x, y), linetype = 3, linewidth=0.8, alpha = 0.4) +
  geom_line(data=data.frame(x=c(1.3, 1.3), y=c(-0.01, 0.01)), 
            aes(x, y), linetype = 3, linewidth=0.8, alpha = 0.4) +
  geom_line(data=data.frame(x=c(-0.1, 2), y=c(0, 0)), 
            aes(x, y), linetype = 3, linewidth=0.8, alpha = 0.4) +
  
  # add data
  geom_line(data=d_conf_p, 
            aes(x=time, 
                y=Coherence, 
                color=Condition),
            linewidth=1) +
  geom_ribbon(data=d_conf_p, 
              aes(x=time, y=Coherence, 
                  ymin=Coherence-SE_Coherence, ymax=Coherence+SE_Coherence,
                  fill=Condition), 
              linetype=0, colour=NA, alpha=0.1, show.legend = F) +
  
  # add significance lines for t-tests
  geom_line(data=rslts_conf_pval,
            aes(x=time,
                y=siglines,
                color=Condition),
            linewidth=1) +
  geom_line(data=rslts_confav_pval,
            aes(x=time,
                y=siglines),
            linewidth=1) +

  # customise
  scale_y_continuous(name='Coherence diff. (HPL - LPL)',
                     breaks=c(-0.02, -0.01, 0, 0.01, 0.02),
                     labels=c('-0.02', '-0.01', '0.00', '0.01', '0.02'),
                     limits=c(-0.035, 0.035),
                     expand=expansion(mult = 0.1)) +
  scale_x_continuous(name='Time from placeholder onset (s)',
                     breaks=c(0, 0.5, 1.0, 1.5, 2.0),
                     labels=c('0.0', '0.5', '1.0', '1.5', '2.0')) +
  scale_color_manual(name='Condition',
                     breaks=c('HIGH', 'LOW'),
                     values=c('HIGH' = '#9467bd',
                              'LOW' = '#17becf'),
                     labels=c('High', 'Low')) +
  geom_rangeframe(data=data.frame(x=c(0, 2), y=c(-0.02, 0.02)),
                  aes(x, y), size=1, color='black') +
  my_theme() + theme(legend.position = 'top')


fig <- plot_grid(plot_rat, 
                 plot_conf,
                 nrow=2,
                 rel_heights=c(1, 1),
                 align="hv")

svg("fig_coherence_rat-confidence_N=34_tar-long.svg",
    width=9, height=8)
plot(fig)
dev.off()

