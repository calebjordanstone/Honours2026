library(tidyverse)
library(data.table)
library(stringr)
library(magrittr)
library(ggplot2)
library(ggpubr)
library(ggthemes)
library(cowplot)
library(RColorBrewer)
library(rstatix)
library(lme4)
library(lmerTest)
library(car)
library(afex)
library(car)
source(file.path("my_theme.R"))

## Set paths -------------------------------------------------------------------
path_exp <- "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/honours_projects/Honours2026/"
path_data <- paste0(path_exp, 'data/sourcedata')

## Import data -----------------------------------------------------------------
files <- list.files(path_data, pattern = '.*(beh.txt)', recursive = T)
d <- rbindlist(lapply(file.path(path_data, files), fread), fill = T)
log_files <- list.files(path_data, pattern = '.*(log.txt)', recursive = T)

## Find correct RTs ------------------------------------------------------------
d_log <- data.table()
for (sub in 1:40) {
  
  d_log_sub <- fread(file.path(path_data, log_files[sub]))
  d_log_sub$Subject <- paste0("sub-", sprintf("%02d", sub))
  
  start_idx <- which(d_log_sub$V3 == 'START_EXP')
  d_log_sub <- d_log_sub[start_idx:nrow(d_log_sub)]
  d_log_sub <- d_log_sub[V3 %in% c('START_TAR', 'RESPONSE')]
  d_log_sub <- d_log_sub[, rt := V1 - lag(V1)]
  d_log_sub <- d_log_sub[V3 == 'RESPONSE', ]
  d_log <- rbind(d_log, d_log_sub)
}

## Clean data ------------------------------------------------------------------
# change choice variable
d[ProbeTrial == 1, ChangeTrial := lag(Choice) != Choice, by=Subject]
d[ProbeTrial == 1, TrialDiff := RunningTrialNo - lag(RunningTrialNo), by=Subject]
d[ProbeTrial == 1, TrialDiffHalf:=ceiling((RunningTrialNo-lag(RunningTrialNo))/2), by=Subject]
d[ProbeTrial == 1, StartTrial:=RunningTrialNo-TrialDiffHalf, by=Subject]
d<-d[order(Subject, -RunningTrialNo)]

new_choice <- ''
sequence <- ''
reassign_choice <- function(ChangeTrial, RunningTrialNo, StartTrial, Choice) {
  if ((is.na(ChangeTrial)) | (ChangeTrial == F)) {
    sequence <<- sequence
  }
  else if (ChangeTrial == T) {
    sequence <<- seq(RunningTrialNo, StartTrial)
    new_choice <<- Choice
  }
  
  if (RunningTrialNo %in% sequence) {
    Choice2 <- new_choice
  }
  else {
    Choice2 <- Choice
  }
  return (Choice2)
}

# d[, Choice2 := reassign_choice(ChangeTrial, RunningTrialNo, StartTrial, Choice),
#     by=.I]
d <- as.data.frame(d)
d %<>%
  rowwise() %>%
  mutate(Choice2=reassign_choice(ChangeTrial, RunningTrialNo, StartTrial, Choice)) %>%
  ungroup()
d <- as.data.table(d)
d<-d[order(Subject, RunningTrialNo)]

# export for Python
#write_csv(d, 'Honours2026_allBeh_Choice2.csv')

# check exclusions 
print(paste("% missing responses: ", d[Response == 999, .N]/nrow(d)*100))
print(paste("Range missing confidence trials: ", 
            d[is.na(Confidence), .N, by=Subject][, min(N)], 
            "-",
            d[is.na(Confidence), .N, by=Subject][, max(N)]))

# exclude missing trials
d <- d[Response != 999, ]
# add new RT data in
d$rt <- d_log$rt
# exclude fast responses
print(paste("% RTs < 200 ms: ", d[rt < .200, .N]/nrow(d)*100))
d <- d[rt >= .200, ]
# exclude poor subjects
exclude <- d[, mean(Accuracy, na.rm=T), by=Subject][V1 < 0.55, Subject]
d <- d[!(Subject %in% exclude), ] 

# add new variables
d <- d[, ':=' (State = case_when(TargetRatio=="[0.9, 0.1]" ~ 0.9,
                                 TargetRatio=="[0.8, 0.2]" ~ 0.8,
                                 TargetRatio=="[0.7, 0.3]" ~ 0.7,
                                 .default= 0.6),
               TargetSideNumeric = ifelse(TargetSide == 'left', 1, 0), 
               ConfidenceTfmd = 0.5 + 0.5*Confidence,
               ProbabilityTfmd = 0.5 + 0.5*Probability,
               ChoiceNumeric = case_when(ProbeTrial == 1 & Choice == 'left' ~ 1.05, 
                                         ProbeTrial == 1 & Choice == 'right' ~ -0.05,
                                         .default = NA),
               Error = case_when(Accuracy == 1 ~ 0,
                                 Accuracy == 0 ~ 1,
                                 .default = Accuracy))]

d <- d[, ':=' (PTarLeft = ifelse(HPL == 'left', State, 1 - State),
               ProbLeft = ifelse(Choice == 'left', ProbabilityTfmd, 1 - ProbabilityTfmd),
               ConfLeft = ifelse(Choice == 'left', ConfidenceTfmd, 1 - ConfidenceTfmd),
               # Ratio = ifelse(State %in% c(0.9, 0.8), 'high', 'low'),
               # Ratio4 = case_when(State == 0.9  ~ 'high',
               #                   State == 0.8 ~ 'med-high',
               #                   State == 0.7 ~ 'med-low',
               #                   State == 0.6 ~ 'low'),
               # Experience = ifelse(StateTrialNo <= 60, 'low', 'high'), # 'low', 'high'
               TargetChoiceCongruency = case_when(TargetSide == 'right' & Choice2 == 'right' ~ "Congruent",
                                                  TargetSide == 'right' & Choice2 == 'left' ~ "Incongruent",
                                                  TargetSide == 'left' & Choice2 == 'right' ~ "Incongruent",
                                                  TargetSide == 'left' & Choice2 == 'left' ~ "Congruent",
                                                  .default = NA),
               ChoiceHPLCongruency = case_when(Choice2 == 'right' & HPL == 'right' ~ "Congruent",
                                               Choice2 == 'right' & HPL == 'left' ~ "Incongruent",
                                               Choice2 == 'left' & HPL == 'right' ~ "Incongruent",
                                               Choice2 == 'left' & HPL == 'left' ~ "Congruent",
                                                  .default = NA),
               TargetHPLCongruency = ifelse(TargetSide == HPL, 'Congruent', 'Incongruent'))] #"Congruent", "Incongruent"

# d <- d[, Congruency := case_when(TargetChoiceCongruency == 'Congruent' & TargetHPLCongruency == 'Congruent' ~ 'subj-con_obj-con',
#                                  TargetChoiceCongruency == 'Congruent' & TargetHPLCongruency == 'Incongruent' ~ 'subj-con_obj-incon',
#                                  TargetChoiceCongruency == 'Incongruent' & TargetHPLCongruency == 'Congruent' ~ 'subj-incon_obj-con',
#                                  TargetChoiceCongruency == 'Incongruent' & TargetHPLCongruency == 'Incongruent' ~ 'subj-incon_obj-incon',
#                                  .default = NA)]

d <- d[ProbeTrial == 1, 
       ChoiceAccuracy := ifelse(HPL == Choice2, 1, 0)]

# d <- d[, Condition := case_when(Ratio == 'high' & Experience == 'high' ~ 'HR_HE',
#                                 Ratio == 'high' & Experience == 'low' ~ 'HR_LE',
#                                 Ratio == 'low' & Experience == 'high' ~ 'LR_HE',
#                                 Ratio == 'low' & Experience == 'low' ~ 'LR_LE')]

d <- d[, ':=' (RollingAverage = frollmean(TargetSideNumeric, n=8, algo='exact'),
               ConfidenceZ = scale(ConfidenceTfmd),
               ProbabilityZ = scale(ProbabilityTfmd)), 
       by=Subject]

# d <- d[, Condition2 := case_when(ConfidenceZ >= 0 & ProbabilityZ >= 0 ~ 'HC_HP',
#                                  ConfidenceZ < 0  & ProbabilityZ >= 0 ~ 'LC_HP',
#                                  ConfidenceZ >= 0 & ProbabilityZ < 0 ~ 'HC_LP',
#                                  ConfidenceZ < 0 & ProbabilityZ < 0 ~ 'LC_LP')]

d <- d[, ':=' (ConfidenceBin = ifelse(ConfidenceZ >= 0, 'high', 'low'),
               ProbabilityBin = ifelse(ProbabilityZ >= 0, 'high', 'low'))]

d <- d[, State2 := case_when(State == 0.9 & TargetHPLCongruency == 'Congruent' ~ 0.9,
                             State == 0.9 & TargetHPLCongruency == 'Incongruent' ~ 0.1,
                             State == 0.8 & TargetHPLCongruency == 'Congruent' ~ 0.8,
                             State == 0.8 & TargetHPLCongruency == 'Incongruent' ~ 0.2,
                             State == 0.7 & TargetHPLCongruency == 'Congruent' ~ 0.7,
                             State == 0.7 & TargetHPLCongruency == 'Incongruent' ~ 0.3,
                             State == 0.6 & TargetHPLCongruency == 'Congruent' ~ 0.6,
                             State == 0.6 & TargetHPLCongruency == 'Incongruent' ~ 0.4
                             )]

d <- d[, SubjProbTarApp := ifelse(TargetSide == Choice2, ProbabilityTfmd, 1 - ProbabilityTfmd)]
d[, new_block := ifelse(lag(Block) != Block, 1, 0), by=Subject]
d[, after_probe := ifelse(lag(ProbeTrial) == 1, 1, 0), by=Subject]
d[, probe_or_block := case_when(new_block == 1 ~ 1, 
                                after_probe == 1 ~ 1,
                                #is.na(new_block) ~ 0,
                                .default = 0), by=Subject]
d[, incongruentState := ifelse(State2 %in% c(0.1, 0.2, 0.3, 0.4), 1, 0)]


### Analysis and Plotting ### --------------------------------------------------
## Analyse meta-cognitive data -------------------------------------------------
# Probability
probs <- d[!(is.na(ProbabilityTfmd)), 
           .(AverageProbabilityRating=mean(ProbabilityTfmd)), 
           by=c('Subject', 'State')]

# run stats
probs[, state := scale(State, center=T, scale=F)]
mdl_probs <- lmer(AverageProbabilityRating ~ 1 + state + (1 + state|Subject), data=probs)
summary(mdl_probs)

# plot
probs$State <-as.factor(probs$State)
probs_av <- probs[, 
                  .(AverageProbabilityRating = mean(AverageProbabilityRating),
                    sdProbabilityRating = sd(AverageProbabilityRating),
                    seProbabilityRating = sd(AverageProbabilityRating)/sqrt(.N)), 
                  by=c('State')]
#cols_prob <- brewer.pal(9, "Blues")[5:9]
cols_prob <- brewer.pal(9, "Blues")[9]

p_prob <- ggplot() +
  
  #add reference line at zero
  geom_line(data=data.frame(x=c('0.6', '0.7', '0.8', '0.9'), y=c(0.6, 0.7, 0.8, 0.9)),
            aes(x, y, group=1), linetype = 3, linewidth=0.8, alpha = 1) +
  
  # geom_point(data=probs,
  #            aes(x=State,
  #                y=AverageProbabilityRating,
  #                color=State),
  #            position=position_jitter(width=0.075),
  #            alpha=0.5) +
  
  geom_pointrange(data=probs_av,
                  aes(x=State,
                      y=AverageProbabilityRating,
                      ymin=AverageProbabilityRating-sdProbabilityRating,
                      ymax=AverageProbabilityRating+sdProbabilityRating),
                  color=cols_prob,
                  linewidth=1.25,
                  position=position_dodge(width=0.4)) +
  
  geom_line(data=probs_av,
             aes(x=State,
                 y=AverageProbabilityRating,
                 group=1),
            size=1,
            linetype=2) + 
  geom_point(data=probs_av,
             aes(x=State,
                 y=AverageProbabilityRating),
             color=cols_prob,
             shape=23,
             fill='white',
             size=4,
             stroke=2) +
  scale_y_continuous(name="Probability Rating",
                     limits=c(0.5, 1),
                     breaks=c(0.5, 0.6, 0.7, 0.8, 0.9, 1)) +
  # scale_y_continuous(name="Probability Rating",
  #                    limits=c(0.8, 0.9), 
  #                    breaks=c(0.8, 0.85, 0.9)) + 
  scale_x_discrete(name="State",
                   breaks=c(0.6, 0.7, 0.8, 0.9)) +
  #scale_color_manual(values = cols_prob) +
  geom_rangeframe(data=data.frame(x=c('0.6', '0.9'), y=c(0.5, 1)),
                  aes(x, y), size=1, color='black') +
  my_theme()


# Confidence
conf <- d[!(is.na(ConfidenceTfmd)), 
          .(AverageConfidenceRating=mean(ConfidenceTfmd)), 
          by=c('Subject', 'State', 'ChoiceHPLCongruency')]

# run stats
conf[, state := scale(State, center=T, scale=F)]
mdl_conf <- lmer(AverageConfidenceRating ~ 1 + state*ChoiceHPLCongruency + (1 + state*ChoiceHPLCongruency|Subject), data=conf)
summary(mdl_conf)


# plot
conf$State <-as.factor(conf$State)
conf$ChoiceHPLCongruency <-as.factor(conf$ChoiceHPLCongruency)

conf_av <- conf[, 
                .(AverageConfidenceRating = mean(AverageConfidenceRating),
                  sdConfidenceRating = sd(AverageConfidenceRating),
                  seConfidenceRating = sd(AverageConfidenceRating)/sqrt(.N)), 
                  by=c('State', 'ChoiceHPLCongruency')]

cols <- brewer.pal(9, "Blues")
cols_conf <- c(cols[9], cols[4])

p_conf <- ggplot() +
  # geom_point(data=conf,
  #            aes(x=State,
  #                y=AverageConfidenceRating,
  #                color=ChoiceHPLCongruency),
  #            position=position_jitterdodge(jitter.width=0.25,
  #                                          dodge.width=0.4),
  #            alpha=0.5) +
  geom_pointrange(data=conf_av,
                  aes(x=State,
                      y=AverageConfidenceRating,
                      ymin=AverageConfidenceRating-sdConfidenceRating,
                      ymax=AverageConfidenceRating+sdConfidenceRating,
                      color=ChoiceHPLCongruency),
                  linewidth=1.25,
                  position=position_dodge(width=0.4)) +
  geom_line(data=conf_av,
            aes(x=State,
                y=AverageConfidenceRating,
                group=ChoiceHPLCongruency),
            position=position_dodge(width=0.4),
            size=1,
            linetype=2) + 
  geom_point(data=conf_av,
             aes(x=State,
                 y=AverageConfidenceRating,
                 color=ChoiceHPLCongruency),
             position=position_dodge(width=0.4),
             shape=23,
             fill='white',
             size=4,
             stroke=2) + 
  scale_y_continuous(name="Confidence Rating",
                     limits=c(0.5, 1),
                     breaks=c(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) +

  # scale_y_continuous(name="Confidence Rating",
  #                    limits=c(0.8, 0.9), 
  #                    breaks=c(0.8, 0.85, 0.9)) + 
  
  scale_x_discrete(name="State",
                     breaks=c(0.6, 0.7, 0.8, 0.9)) +
  geom_rangeframe(data=data.frame(x=c('0.6', '0.9'), y=c(0.5, 1)),
                  aes(x, y), size=1, color='black') +
  scale_color_manual(values = cols_conf) +
  my_theme() +
  theme(legend.position = 'top')

# Choice
choice <- d[!(is.na(ConfidenceTfmd)), 
          .(AverageChoiceAccuracy=mean(ChoiceAccuracy, na.rm=T)), 
          by=c('Subject', 'State')]


# run stats
choice[, state := scale(State, center=T, scale=F)]
mdl_choice <- lmer(AverageChoiceAccuracy ~ 1 + state + (1|Subject), data=choice)
summary(mdl_choice)



choice$State <- as.factor(choice$State)

choice_av <- choice[, 
                .(AverageChoiceAccuracy = mean(AverageChoiceAccuracy),
                  sdChoiceAccuracy = sd(AverageChoiceAccuracy),
                  seChoiceAccuracy = sd(AverageChoiceAccuracy)/sqrt(.N)), 
                by=c('State')]

cols_choice <- brewer.pal(9, "Blues")[5:9]
cols_choice <- brewer.pal(9, "Blues")[9]
p_choice <- ggplot() +
  
  # add reference line at zero
  geom_line(data=data.frame(x=c('0.6', '0.9'), y=c(0.5, 0.5)),
            aes(x, y, group=1), linetype = 3, linewidth=0.8, alpha = 1) +

  # geom_point(data=choice,
  #            aes(x=State,
  #                y=AverageChoiceAccuracy,
  #                color=State),
  #            position=position_jitter(width=0.075),
  #            alpha=0.5) +
  # 
  geom_pointrange(data=choice_av,
                  aes(x=State,
                      y=AverageChoiceAccuracy,
                      ymax=AverageChoiceAccuracy+sdChoiceAccuracy,
                      ymin=AverageChoiceAccuracy-sdChoiceAccuracy),
                  color=cols_choice,
                  linewidth=1.25) +
  geom_line(data=choice_av,
            aes(x=State,
                y=AverageChoiceAccuracy,
                group=1),
            size=1,
            linetype=2) + 
  geom_point(data=choice_av,
             aes(x=State,
                 y=AverageChoiceAccuracy),
             color=cols_choice,
             shape=23,
             fill='white',
             size=4,
             stroke=2) + 
  scale_y_continuous(name="Proportion of Correct Choices",
                     limits=c(0, 1),
                     breaks=c(0, 0.2, 0.4, 0.6, 0.8, 1.0)) +
  # scale_y_continuous(name="Proportion of Correct Choices",
  #                    limits=c(0.4, 0.7), 
  #                    breaks=c(0.4, 0.5, 0.6, 0.7)) + 
  scale_x_discrete(name="State",
                     breaks=c(0.6, 0.7, 0.8, 0.9)) +
  geom_rangeframe(data=data.frame(x=c('0.6', '0.9'), y=c(0, 1)),
                  aes(x, y), size=1, color='black') +
  #scale_color_manual(values = cols_choice) +
  my_theme()


fig <- plot_grid(p_choice, 
                 p_conf,
                 p_prob,
                 nrow=1,
                 rel_heights=c(1, 1),
                 align="hv")

# save figure
svg("fig_metacognition_gav_sd.svg",
    width=10, height=4)
plot(fig)
dev.off()


## Analyse RT and error rate data -------------------------------------------------

# Response time
rt <- d[,
        .(AverageRT=mean(rt)), 
           by=c('Subject', 'State', 'TargetHPLCongruency')]

# run stats
rt[, state := scale(State, center=T, scale=F)]
mdl_rt <- lmer(AverageRT ~ 1 + state*TargetHPLCongruency + (1+state*TargetHPLCongruency|Subject), data=rt)
summary(mdl_rt)

# plot
rt$State <-as.factor(rt$State)
rt$TargetHPLCongruency <-as.factor(rt$TargetHPLCongruency)

rt_av <- rt[, 
            .(AverageRT = mean(AverageRT),
              seRT=sd(AverageRT)/sqrt(.N)),
            by=c('State', 'TargetHPLCongruency')]



cols <- brewer.pal(9, "Blues")
cols_rt <- c(cols[9], cols[4])

p_rt <- ggplot() +
  
  # geom_point(data=rt,
  #            aes(x=State,
  #                y=AverageRT,
  #                color=TargetHPLCongruency),
  #            position=position_jitterdodge(jitter.width=0.25,
  #                                          dodge.width=0.4),
  #            alpha=0.5) +
  
  geom_line(data=rt_av,
            aes(x=State,
                y=AverageRT,
                group=TargetHPLCongruency),
            position=position_dodge(width=0.4),
            size=1,
            linetype=2) + 
  geom_pointrange(data=rt_av,
                  aes(x=State,
                      y=AverageRT,
                      ymin=AverageRT-seRT,
                      ymax=AverageRT+seRT,
                      color=TargetHPLCongruency),
                  linewidth=1.25,
                  position=position_dodge(width=0.4)) +
  geom_point(data=rt_av,
             aes(x=State,
                 y=AverageRT,
                 color=TargetHPLCongruency),
             position=position_dodge(width=0.4),
             shape=23,
             fill='white',
             size=4,
             stroke=2) + 
  

  # scale_y_continuous(name="Reaction time (s)",
  #                    limits=c(0.3, 0.8), 
  #                    breaks=c(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) +
  scale_y_continuous(name="Reaction time (s)",
                     limits=c(0.45, 0.6), 
                     breaks=c(0.45, 0.5, 0.55, 0.6)) +
  
  scale_x_discrete(name="State",
                   breaks=c(0.6, 0.7, 0.8, 0.9)) +
  scale_color_manual(values = cols_rt) +
  geom_rangeframe(data=data.frame(x=c('0.6', '0.9'), y=c(0.45, 0.6)),
                  aes(x, y), size=1, color='black') +
  my_theme() +
  theme(legend.position = 'top')


# Error rate
er <- d[,
        .(AverageER=mean(Error)), 
        by=c('Subject', 'State', 'TargetHPLCongruency')]

# run stats
er[, state := scale(State, center=T, scale=F)]
mdl_er <- lmer(AverageER ~ 1 + state*TargetHPLCongruency + (1+state+TargetHPLCongruency|Subject), data=er)
summary(mdl_er)


# plot
er$State <-as.factor(er$State)
er$TargetHPLCongruency <-as.factor(er$TargetHPLCongruency)

er_av <- er[, 
            .(AverageER = mean(AverageER),
              seER=sd(AverageER)/sqrt(.N)),
            by=c('State', 'TargetHPLCongruency')]



cols <- brewer.pal(9, "Blues")
cols_er <- c(cols[9], cols[4])

p_er <- ggplot() +
  
  # geom_point(data=er,
  #            aes(x=State,
  #                y=AverageER,
  #                color=TargetHPLCongruency),
  #            position=position_jitterdodge(jitter.width=0.25,
  #                                          dodge.width=0.4),
  #            alpha=0.5) +
  
  geom_line(data=er_av,
            aes(x=State,
                y=AverageER,
                group=TargetHPLCongruency),
            position=position_dodge(width=0.4),
            size=1,
            linetype=2) + 
  geom_pointrange(data=er_av,
                  aes(x=State,
                      y=AverageER,
                      ymin=AverageER-seER,
                      ymax=AverageER+seER,
                      color=TargetHPLCongruency),
                  linewidth=1.25,
                  position=position_dodge(width=0.4)) +
  geom_point(data=er_av,
             aes(x=State,
                 y=AverageER,
                 color=TargetHPLCongruency),
             position=position_dodge(width=0.4),
             shape=23,
             fill='white',
             size=4,
             stroke=2) + 
  
  
  # scale_y_continuous(name="Reaction time (s)",
  #                    limits=c(0.3, 0.8), 
  #                    breaks=c(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) +
  scale_y_continuous(name="Error rate",
                     limits=c(0.1, 0.4), 
                     breaks=c(0.1, 0.2, 0.3, 0.4)) +
  
  scale_x_discrete(name="State",
                   breaks=c(0.6, 0.7, 0.8, 0.9)) +
  scale_color_manual(values = cols_er) +
  geom_rangeframe(data=data.frame(x=c('0.6', '0.9'), y=c(0.1, 0.4)),
                  aes(x, y), size=1, color='black') +
  my_theme() +
  theme(legend.position = 'top')



fig <- plot_grid(p_rt, 
                 p_er,
                 nrow=1,
                 rel_heights=c(1, 1),
                 align="hv")

# save figure
svg("fig_beh.svg",
    width=7.5, height=4)
plot(fig)
dev.off()


## Analyse RT modulation by confidence --------------------------------------------
x <- d[!is.na(Confidence), 
       .(Subject, 
         TargetHPLCongruency, 
         ChoiceHPLCongruency, 
         probe_or_block, 
         RunningTrialNo, 
         State, 
         ProbabilityTfmd, 
         ConfidenceTfmd, 
         rt,
         Error)]
x[, ':=' (sqrt_rt = sqrt(rt),
          log_rtn = log(RunningTrialNo),
          logit_state = logit(State))]
x[, ':=' (ctrd_rtn = scale(log_rtn, center=T, scale=F), 
          ctrd_state = scale(logit_state, center=T, scale=F),
          ctrd_prob = scale(ProbabilityTfmd, center=T, scale=F),
          ctrd_conf = scale(ConfidenceTfmd, center=T, scale=F)),
  by='Subject']


mdl1 <- lmer(sqrt_rt ~
              probe_or_block + 
              ctrd_rtn +
              ctrd_state*TargetHPLCongruency + 
              ctrd_state:TargetHPLCongruency:ChoiceHPLCongruency:ctrd_prob +
              ctrd_state:TargetHPLCongruency:ChoiceHPLCongruency:ctrd_conf +
              (0 + 
                 probe_or_block + # parameter with the 5th variance in the full model
                 ctrd_rtn + # parameter with the 2nd least variance in the full model
                 ctrd_state +  # parameter with the least variance in the full model
                 TargetHPLCongruency|Subject) + # parameter with the 3rd variance in the full model
                 #ctrd_state:TargetHPLCongruency + # parameter with the 4th variance in the full model
                 #ctrd_state:TargetHPLCongruency:ChoiceHPLCongruency:ctrd_prob|Subject) +
                 #ctrd_state:TargetHPLCongruency:ChoiceHPLCongruency:ctrd_conf|Subject) + 
              (1|Subject),
            data=x,
            control = lmerControl(calc.derivs=F))
summary(mdl1)
plot(mdl1)
hist(residuals(mdl1))

x <- data.frame(coef(summary(mdl1)))
write.csv(x, "C:\\Users\\cstone\\OneDrive - UNSW\\Documents\\Projects\\honours_projects\\Honours2026\\output\\mdl1_rt.csv", row.names = T)



mdl2 <- glmer(Error ~
                probe_or_block + # removed these because they are more relevant to RT I think?
                ctrd_rtn +
                ctrd_state*
                TargetHPLCongruency + 
                ctrd_state:TargetHPLCongruency:ChoiceHPLCongruency:ctrd_prob +
                ctrd_state:TargetHPLCongruency:ChoiceHPLCongruency:ctrd_conf +
                (0 + 
                   probe_or_block + 
                   ctrd_rtn + 
                   ctrd_state* # 2nd least variance, after itneraction effect
                   TargetHPLCongruency|Subject) + # 3rd least variance 
                   #ctrd_state:TargetHPLCongruency:ChoiceHPLCongruency:ctrd_prob|Subject) +
                   #ctrd_state:TargetHPLCongruency:ChoiceHPLCongruency:ctrd_conf|Subject) +
                (1|Subject),
              data=x,
              family = binomial,
              control = glmerControl(calc.derivs=F))

summary(mdl2)

x <- data.frame(coef(summary(mdl2)))
write.csv(x, "C:\\Users\\cstone\\OneDrive - UNSW\\Documents\\Projects\\honours_projects\\Honours2026\\output\\mdl2_er.csv", row.names = T)


## Plot
# create confidence and probability bins
d[!(is.na(Confidence)), ':='
  (Probability_zscore = scale(ProbabilityTfmd), 
    Confidence_zscore = scale(ConfidenceTfmd)),
    by=c('Subject')]
d[!(is.na(Confidence)), ':='
  (ProbabilityBin_zscore = ifelse(Probability_zscore >= 0, 'high', 'low'), 
    ConfidenceBin_zscore = ifelse(Confidence_zscore >= 0, 'high', 'low')),
  by=c('Subject')]

## Plot difference scores 
# Response time
# rt_diff <- d[!(is.na(Confidence)),
#              .(AverageRT=mean(rt),
#                TrialCounts=.N), 
#              by=c('Subject', 'State', 'ConfidenceBin_zscore', 'TargetHPLCongruency', 'ChoiceHPLCongruency')]
# rt_diff$State <- as.factor(rt_diff$State)
# rt_diff$ConfidenceBin_zscore <- as.factor(rt_diff$ConfidenceBin_zscore)
# rt_diff$TargetHPLCongruency <- as.factor(rt_diff$TargetHPLCongruency)
# rt_diff$ChoiceHPLCongruency <- as.factor(rt_diff$ChoiceHPLCongruency)
# 
# 
# rt_diff <- dcast.data.table(rt_diff, 
#                         Subject + State + ConfidenceBin_zscore + ChoiceHPLCongruency ~ TargetHPLCongruency, 
#                         value.var=c('AverageRT'))
# 
# rt_diff <- rt_diff[, .(RTdiff = Incongruent - Congruent),
#            by = c('Subject', 'State', 'ConfidenceBin_zscore', 'ChoiceHPLCongruency')]
# 
# 
# rt_diff_av <- rt_diff[, 
#             .(AverageRTDiff = mean(RTdiff, na.rm=T),
#               seRTDiff=sd(RTdiff, na.rm=T)/sqrt(.N)),
#             by=c('State', 'ConfidenceBin_zscore', 'ChoiceHPLCongruency')]
# 
# 
# 
# cols <- brewer.pal(9, "Blues")
# cols_rt <- c(cols[9], cols[4])
# 
# p_rt_diff <- ggplot() +
#   
#   # add reference line at zero
#   geom_line(data=data.frame(x=c('0.6', '0.9'), y=c(0, 0)),
#             aes(x, y, group=1), linetype = 3, linewidth=0.8, alpha = 1) +
#   
#   # geom_point(data=rt,
#   #            aes(x=State,
#   #                y=AverageRT,
#   #                color=TargetHPLCongruency),
#   #            position=position_jitterdodge(jitter.width=0.25,
#   #                                          dodge.width=0.4),
#   #            alpha=0.5) +
#   
#   geom_line(data=rt_diff_av,
#             aes(x=State,
#                 y=AverageRTDiff,
#                 group=ConfidenceBin_zscore),
#             position=position_dodge(width=0.4),
#             size=1,
#             linetype=2) + 
#   geom_pointrange(data=rt_diff_av,
#                   aes(x=State,
#                       y=AverageRTDiff,
#                       ymin=AverageRTDiff-seRTDiff,
#                       ymax=AverageRTDiff+seRTDiff,
#                       color=ConfidenceBin_zscore),
#                   linewidth=1.25,
#                   position=position_dodge(width=0.4)) +
#   geom_point(data=rt_diff_av,
#              aes(x=State,
#                  y=AverageRTDiff,
#                  color=ConfidenceBin_zscore),
#              position=position_dodge(width=0.4),
#              shape=23,
#              fill='white',
#              size=4,
#              stroke=2) + 
#   facet_wrap('ChoiceHPLCongruency') +
#   
#   
#   # scale_y_continuous(name="Reaction time (s)",
#   #                    limits=c(0.3, 0.8), 
#   #                    breaks=c(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) +
#   scale_y_continuous(name="RT diff. (Incongruent - Congruent) (s)",
#                      limits=c(-0.015, 0.085), 
#                      breaks=c(0, 0.02, 0.04, 0.06, 0.08)) +
#   
#   scale_x_discrete(name="State",
#                    breaks=c(0.6, 0.7, 0.8, 0.9)) +
#   scale_color_manual(values = cols_rt) +
#   geom_rangeframe(data=data.frame(x=c('0.6', '0.9'), y=c(0, 0.08)),
#                   aes(x, y), size=1, color='black') +
#   my_theme() +
#   theme(legend.position = 'top')
# 
# 
# # Error rate
# er_diff <- d[!(is.na(Confidence)),
#              .(AverageER=mean(Error),
#                TrialCounts=.N), 
#              by=c('Subject', 'State', 'ConfidenceBin_zscore', 'TargetHPLCongruency', 'ChoiceHPLCongruency')]
# er_diff$State <- as.factor(er_diff$State)
# er_diff$ConfidenceBin_zscore <- as.factor(er_diff$ConfidenceBin_zscore)
# er_diff$TargetHPLCongruency <- as.factor(er_diff$TargetHPLCongruency)
# er_diff$ChoiceHPLCongruency <- as.factor(er_diff$ChoiceHPLCongruency)
# 
# 
# er_diff <- dcast.data.table(er_diff, 
#                             Subject + State + ConfidenceBin_zscore + ChoiceHPLCongruency ~ TargetHPLCongruency, 
#                             value.var=c('AverageER'))
# 
# er_diff <- er_diff[, .(ERdiff = Incongruent - Congruent),
#                    by = c('Subject', 'State', 'ConfidenceBin_zscore', 'ChoiceHPLCongruency')]
# 
# 
# er_diff_av <- er_diff[, 
#                       .(AverageERDiff = mean(ERdiff, na.rm=T),
#                         seERDiff=sd(ERdiff, na.rm=T)/sqrt(.N)),
#                       by=c('State', 'ConfidenceBin_zscore', 'ChoiceHPLCongruency')]
# 
# 
# cols <- brewer.pal(9, "Blues")
# cols_er <- c(cols[9], cols[4])
# 
# p_er_diff <- ggplot() +
#   
#   # add reference line at zero
#   geom_line(data=data.frame(x=c('0.6', '0.9'), y=c(0, 0)),
#             aes(x, y, group=1), linetype = 3, linewidth=0.8, alpha = 1) +
#   
#   # geom_point(data=rt,
#   #            aes(x=State,
#   #                y=AverageRT,
#   #                color=TargetHPLCongruency),
#   #            position=position_jitterdodge(jitter.width=0.25,
#   #                                          dodge.width=0.4),
#   #            alpha=0.5) +
#   
#   geom_line(data=er_diff_av,
#             aes(x=State,
#                 y=AverageERDiff,
#                 group=ConfidenceBin_zscore),
#             position=position_dodge(width=0.4),
#             size=1,
#             linetype=2) + 
#   geom_pointrange(data=er_diff_av,
#                   aes(x=State,
#                       y=AverageERDiff,
#                       ymin=AverageERDiff-seERDiff,
#                       ymax=AverageERDiff+seERDiff,
#                       color=ConfidenceBin_zscore),
#                   linewidth=1.25,
#                   position=position_dodge(width=0.4)) +
#   geom_point(data=er_diff_av,
#              aes(x=State,
#                  y=AverageERDiff,
#                  color=ConfidenceBin_zscore),
#              position=position_dodge(width=0.4),
#              shape=23,
#              fill='white',
#              size=4,
#              stroke=2) + 
#   facet_wrap('ChoiceHPLCongruency') +
#   
#   # scale_y_continuous(name="Reaction time (s)",
#   #                    limits=c(0.3, 0.8), 
#   #                    breaks=c(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) +
#   scale_y_continuous(name="Error diff. (Incongruent - Congruent)",
#                      limits=c(-0.15, 0.3), 
#                      breaks=c(-0.1, -0.1, 0, 0.1, 0.2, 0.3)) +
#   
#   scale_x_discrete(name="State",
#                    breaks=c(0.6, 0.7, 0.8, 0.9)) +
#   scale_color_manual(values = cols_rt) +
#   geom_rangeframe(data=data.frame(x=c('0.6', '0.9'), y=c(-0.1, 0.3)),
#                   aes(x, y), size=1, color='black') +
#   my_theme() +
#   theme(legend.position = 'top')
# 
# 
# fig <- plot_grid(p_rt_diff, 
#                  p_er_diff,
#                  nrow=2,
#                  rel_heights=c(1, 1),
#                  align="hv")
# 
# # save figure
# svg("fig_beh.svg",
#     width=7.5, height=9)
# plot(fig)
# dev.off()
# 



## Plot significant effects from models
er_conf <- d[!(is.na(Confidence)),
        .(AverageER=mean(Error)), 
         by=c('Subject', 'State', 'ConfidenceBin_zscore', 'TargetHPLCongruency', 'ChoiceHPLCongruency')]
er_conf$State <- as.factor(er_conf$State)
er_conf$ConfidenceBin_zscore <- as.factor(er_conf$ConfidenceBin_zscore)
er_conf$TargetHPLCongruency <- as.factor(er_conf$TargetHPLCongruency)
er_conf$ChoiceHPLCongruency <- as.factor(er_conf$ChoiceHPLCongruency)

er_conf_av <- er_conf[, .(MeanER = mean(AverageER),
                seER = sd(AverageER)/sqrt(.N)),
            by=c('State', 'ConfidenceBin_zscore', 'TargetHPLCongruency', 'ChoiceHPLCongruency')]
er_conf_av <- er_conf_av[TargetHPLCongruency == 'Congruent' & ChoiceHPLCongruency == 'Congruent', ]

cols <- brewer.pal(9, "Blues")
cols_er <- c(cols[9], cols[4])

p_er_conf <- ggplot() +

  
  geom_line(data=er_conf_av,
            aes(x=State,
                y=MeanER,
                group=ConfidenceBin_zscore),
            position=position_dodge(width=0.4),
            size=1,
            linetype=2) + 
  geom_pointrange(data=er_conf_av,
                  aes(x=State,
                      y=MeanER,
                      ymin=MeanER-seER,
                      ymax=MeanER+seER,
                      color=ConfidenceBin_zscore),
                  linewidth=1.25,
                  position=position_dodge(width=0.4)) +
  geom_point(data=er_conf_av,
             aes(x=State,
                 y=MeanER,
                 color=ConfidenceBin_zscore),
             position=position_dodge(width=0.4),
             shape=23,
             fill='white',
             size=4,
             stroke=2) + 
  #facet_grid(vars(ChoiceHPLCongruency), vars(TargetHPLCongruency)) +
  
  # scale_y_continuous(name="Reaction time (s)",
  #                    limits=c(0.3, 0.8), 
  scale_y_continuous(name="Error rate",
                     limits=c(0, 0.45), 
                     breaks=c(0.1, 0.2, 0.3, 0.4)) +
  scale_x_discrete(name="State",
                   breaks=c(0.6, 0.7, 0.8, 0.9)) +
  scale_color_manual(values = cols_er) +
  geom_rangeframe(data=data.frame(x=c('0.6', '0.9'), y=c(0.1, 0.4)),
                  aes(x, y), size=1, color='black') +
  my_theme() +
  theme(legend.position = 'top')


er_prob <- d[!(is.na(Confidence)),
             .(AverageER=mean(Error)), 
             by=c('Subject', 'State', 'ProbabilityBin_zscore', 'TargetHPLCongruency', 'ChoiceHPLCongruency')]
er_prob$State <- as.factor(er_prob$State)
er_prob$ConfidenceBin_zscore <- as.factor(er_prob$ConfidenceBin_zscore)
er_prob$TargetHPLCongruency <- as.factor(er_prob$TargetHPLCongruency)
er_prob$ChoiceHPLCongruency <- as.factor(er_prob$ChoiceHPLCongruency)

er_prob_av <- er_prob[, .(MeanER = mean(AverageER),
                          seER = sd(AverageER)/sqrt(.N)),
                      by=c('State', 'ProbabilityBin_zscore', 'TargetHPLCongruency', 'ChoiceHPLCongruency')]
er_prob_av <- er_prob_av[TargetHPLCongruency == 'Incongruent' & ChoiceHPLCongruency == 'Congruent', ]

cols <- brewer.pal(9, "Blues")
cols_er <- c(cols[9], cols[4])

p_er_prob <- ggplot() +

  geom_line(data=er_prob_av,
            aes(x=State,
                y=MeanER,
                group=ProbabilityBin_zscore),
            position=position_dodge(width=0.4),
            size=1,
            linetype=2) + 
  geom_pointrange(data=er_prob_av,
                  aes(x=State,
                      y=MeanER,
                      ymin=MeanER-seER,
                      ymax=MeanER+seER,
                      color=ProbabilityBin_zscore),
                  linewidth=1.25,
                  position=position_dodge(width=0.4)) +
  geom_point(data=er_prob_av,
             aes(x=State,
                 y=MeanER,
                 color=ProbabilityBin_zscore),
             position=position_dodge(width=0.4),
             shape=23,
             fill='white',
             size=4,
             stroke=2) + 

  scale_y_continuous(name="Error rate",
                     limits=c(0, 0.45), 
                     breaks=c(0.1, 0.2, 0.3, 0.4)) +
  scale_x_discrete(name="State",
                   breaks=c(0.6, 0.7, 0.8, 0.9)) +
  scale_color_manual(values = cols_er) +
  geom_rangeframe(data=data.frame(x=c('0.6', '0.9'), y=c(0.1, 0.4)),
                  aes(x, y), size=1, color='black') +
  my_theme() +
  theme(legend.position = 'top')


fig <- plot_grid(p_er_conf,
                 p_er_prob,
                 ncol=2,
                 rel_heights=c(1, 1),
                 align="hv")

# save figure
svg("fig_er_sig_effects.svg",
    width=7.5, height=5)
plot(fig)
dev.off()
# 


