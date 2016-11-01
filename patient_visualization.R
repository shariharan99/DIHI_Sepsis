source("/Users/Sanjay/Desktop/Research/DIHI/DIHI_Sepsis/load_data.R")

library(lubridate)
library(stringr)

plotPatientTimeSeries = function(enc_id = NULL){
  if (is.null(enc_id)){
    enc_id = sample(enc$ENCOUNTER_ID, size = 1)
  }
  
  #admit and discharge
  row = enc[ENCOUNTER_ID == enc_id]
  admit_date = row$ADMISSION_DATE_TIME
  dis_date = row$DISCHARGE_DATE_TIME
  timeline = data.table("event" = c("admit", "discharge"),"time" = c(admit_date, dis_date))
  timeline$time = as.POSIXct(timeline$time)
  
  #Meds
  row = med[ENCOUNTER_ID == enc_id]
  names = names(row)
  names = names[3:length(names)]
  for (name in names){
    if (!is.na(row[[name]])) {
      timeline = rbind(timeline, list(strsplit(names[1], "_")[[1]][1], as.POSIXct(row[[name]])))
    }
  }
  
  #Labs
  rows = Labs[ENCOUNTER_ID == enc_id]
  timeline = rbind(timeline, list(rows$LAB_NAME, as.POSIXct(rows$RESULT_TIME, format="%m/%d/%y %H:%M:%S")))
  
  #Reorder and factorize event column
  setcolorder(timeline, c(2,1))
  timeline$event = as.factor(timeline$event)
  levels = levels(timeline$event)
  timeline$event = as.numeric(timeline$event)
  timeline = as.xts.data.table(timeline)
  
  plot(timeline, type = "p", pch = 1, bty='L', main = "Timeline")
  points(timeline, col = timeline$event, pch = 19)
  legend('topright', legend = levels, pch = 19, col = 1:length(timeline$event), cex = 0.5)
  
  print(sprintf("Encounter ID: %s", enc_id))
  
}

plotPatientTimeSeries()




