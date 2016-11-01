#Labs
pre_dir = "/Users/Sanjay/Desktop/Research/DIHI/Data/Labs/"
files = dir(pre_dir)
Labs = data.frame()
for (file in files){
  dat = fread(paste0(pre_dir, file))
  dat$LAB_NAME = dat$LAB_NAME[1] #Make each lab file have one unique lab name
  dat$RESULT_VALUE = as.numeric(dat$RESULT_VALUE)
  dat$RESULT_VALUE = scale(dat$RESULT_VALUE)
  Labs = rbind(Labs, dat)
}

plotLabTimeSeries = function(enc_id = NULL){
  if (is.null(enc_id)){
    enc_id = sample(Labs$ENCOUNTER_ID, 1)
  }
  
  lab = Labs[ENCOUNTER_ID == enc_id]
  timeline = data.table("event" = lab$LAB_NAME, "value" = lab$RESULT_VALUE, "time" = as.POSIXct(lab$RESULT_TIME, format="%m/%d/%y %H:%M:%S"))
  setcolorder(timeline, c(3,1,2))
  timeline$event = as.factor(timeline$event)
  levels = levels(timeline$event)
  timeline$event = as.numeric(timeline$event)
  timeline = as.xts.data.table(timeline)
  
  plot(timeline$value, type = "p", pch = 1, bty='L', main = "Timeline")
  points(timeline$value, col = timeline$event, pch = 19)
  legend('topright', legend = levels, pch = 19, col = 1:length(timeline$event), cex = 0.5)
  
  print(sprintf("Encounter ID: %s", enc_id))
  
}

plotLabTimeSeries()

