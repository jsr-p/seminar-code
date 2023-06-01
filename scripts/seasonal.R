library("dplyr")
library("forecast")
library("readr")
library("magrittr")
library("zoo")
library("ggplot2")
library("fpp3")
library("reshape2")
library("stringr")

# install.packages("forecast")
# install.packages("zoo")
# install.packages("fpp3")
# install.packages("languageserver")

df <- read_csv("data/agg.csv") %>% as_tsibble(index=date)

dfall <- melt(df, id = c("date")) %>% 
  as_tsibble(index=date, key=variable)  %>% 
  model(
    classical_decomposition(value, type = "multiplicative")
  ) %>% 
  components() %>% 
  mutate(dayofweek = wday(date, label = TRUE)) %>% 
  tibble() %>% 
  select(variable, dayofweek, seasonal) %>% 
  group_by(variable, dayofweek) %>%
  summarize(avg_seasonal = mean(seasonal)) %>% 
  mutate(dayofweek = factor(dayofweek, c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")))

plot <- ggplot(dfall, aes(x = dayofweek, y = avg_seasonal, color=variable, group=variable)) +
  geom_line() +
  labs(y = "Seasonal index")
ggsave("output/figs/seasonal_weekly.png", plot = plot)

dfall <- melt(df, id = c("date")) %>% 
  as_tsibble(index=date, key=variable)  %>% 
  model(
    classical_decomposition(value ~ season(12), type = "multiplicative")
  ) %>% 
  components() %>% 
  mutate(month = month(date, label = TRUE))  %>% 
  tibble() %>% 
  select(variable, month, seasonal) %>% 
  group_by(variable, month) %>%
  summarize(avg_seasonal = mean(seasonal))  %>% 
  mutate(month = factor(month, month.abb[c(1:12)]))

plot <- ggplot(dfall, aes(x = month, y = avg_seasonal, color=variable, group=variable)) +
  geom_line() +
  labs(y = "Seasonal index")
ggsave("output/figs/seasonal_monthly.png", plot = plot)
