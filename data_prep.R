##### data_prep.R
#####
##### Clean data into format to be directly loaded into python for modeling scripts
#####
##### Created: 11/7/16
##### Author: Joe Futoma

library(lubridate)
library(stringr)
library(data.table)

##### PATHS

sepsis_folder = "~/Dropbox/research/DIHI_Sepsis/"
setwd(sepsis_folder)

encounters_drop_path = "~/Box Sync/Output Data Files (Dr Armando Bedoya)/Encounters_ToDrop.csv"
encounters_path = "~/Box Sync/Output Data Files (Dr Armando Bedoya)/Sepsis_EncounterDetails_Outcomes.csv"

demographics_path = "~/Box Sync/Output Data Files (Dr Armando Bedoya 2)/sepsis_pt_demographics.csv"
demographics_dupes = "~/Box Sync/Output Data Files (Dr Armando Bedoya 2)/sepsis_pt_demographics_duplicates.csv"

labs_folder = "~/Box Sync/Output Data Files (Dr Armando Bedoya 3)/"
meds_folder = "~/Box Sync/Output Data Files/"

##### get final set of encounters that will be modeled, after exclusions

dat.encounters = fread(encounters_path)
dat.encounters[, ADMISSION_DATE_TIME := as.POSIXct(ADMISSION_DATE_TIME, format="%Y-%m-%d %H:%M:%S")]
dat.encounters[, DISCHARGE_DATE_TIME := as.POSIXct(DISCHARGE_DATE_TIME, format="%Y-%m-%d %H:%M:%S")]

dat.encounters.drop = fread(encounters_drop_path) #drop all admits before 7/1/2014; all in this file
encounter.drop.ids = dat.encounters.drop$ENCOUNTER_ID
rm(dat.encounters.drop)

dat.encounters = dat.encounters[!(ENCOUNTER_ID %in% encounter.drop.ids)]
rm(encounter.drop.ids)           

dat.demog.dupes = fread(demographics_dupes)
demog.dupe.mrns = unique(dat.demog.dupes$PATIENT_MRN)
dat.encounters = dat.encounters[!(PATIENT_MRN %in% demog.dupe.mrns)]
rm(dat.demog.dupes); rm(demog.dupe.mrns)

##### create baseline covariates for each encounter & write out

dat.demog = fread(demographics_path)
setkey(dat.encounters,PATIENT_MRN)
setkey(dat.demog,PATIENT_MRN)

dat.encounters = merge(dat.encounters,dat.demog,all.x=T,all.y=F)
rm(dat.demog)
setkey(dat.encounters,ENCOUNTER_ID)

dat.encounters[, c("CITY_OF_ORIGIN","PATIENT_MRN") := NULL]

dobs = as.Date(dat.encounters$DATE_OF_BIRTH,"%d-%b-%y")
dobs = as.Date(format(dobs,"19%y-%m-%d"))
dat.encounters[, DATE_OF_BIRTH := as.POSIXct(dobs, format="%Y-%m-%d")]
dat.encounters[, BASELINE_AGE := as.numeric(ADMISSION_DATE_TIME - DATE_OF_BIRTH)/365.24]                                               
dat.encounters[, c("DATE_OF_BIRTH") := NULL]

dat.encounters$ADMIT_TRANSFER = 1 #Basically, anything but from home
dat.encounters$ADMIT_TRANSFER[dat.encounters$ADMIT_SOURCE=="Home or Non-Health Care Facility Point of Origin"] = 0
dat.encounters$ADMIT_TRANSFER[dat.encounters$ADMIT_SOURCE==""] = 0
dat.encounters[, c("ADMIT_SOURCE") := NULL]

#dat.encounters$ADMIT_NOT_ELECTIVE = 1 #Anything that was not elective. Treat Emergency and Urgent the same.
#dat.encounters$ADMIT_NOT_ELECTIVE[dat.encounters$ADMIT_TYPE=="Elective"] = 0
dat.encounters[, ADMIT_URGENT := as.numeric(ADMIT_TYPE=="Urgent")]
dat.encounters[, ADMIT_EMERGENCY := as.numeric(ADMIT_TYPE=="Emergency")]
dat.encounters[, c("ADMIT_TYPE") := NULL]

dat.encounters[, SEPSIS_BILLING_CODE := as.numeric(SEPSIS_BILLING_CODE=="YES")]

dat.encounters[, IS_MALE := as.numeric(SEX=="Male")]
dat.encounters[, c("SEX") := NULL]

#For now, remove payor
dat.encounters[, c("PAYOR_NAME") := NULL]

#race.  boil down to black/white? 92% fit this. 3% other, 1.5% asian, ...
dat.encounters$IS_NOT_WHITE = 1
dat.encounters$IS_NOT_WHITE[dat.encounters$RACE=="Caucasian/White"] = 0
dat.encounters[, c("RACE") := NULL]

### WEIRD: some encounters have discharge date *after* admit.
### also some patients have weirdly short LOS...
###
### for now, let's eliminate encounters with LOS < 0 (data issue?), but leave rest

dat.encounters = dat.encounters[DISCHARGE_DATE_TIME>ADMISSION_DATE_TIME,]

dat.encounters.final = copy(dat.encounters)
dat.encounters.final[, LOS := as.numeric(DISCHARGE_DATE_TIME-ADMISSION_DATE_TIME)/60.0] #in hours
dat.encounters.final[, c("ADMISSION_DATE_TIME","DISCHARGE_DATE_TIME","PATIENT_ALIVE_STATUS","DEATH_DATE","ED_ARRVL_DATE_TIME") := NULL]

write.csv(dat.encounters.final,"cleaned_data/encounters_info.csv",row.names=F,quote=F)

dat.admits = dat.encounters[, c("ENCOUNTER_ID","PATIENT_MRN","ADMISSION_DATE_TIME","DISCHARGE_DA"), with=F]


###
### test: quick glm to just this data??
###

 library(AUC)
 library(ROCR)
 
 model = glm(SEPSIS_BILLING_CODE ~ BASELINE_AGE+ADMIT_TRANSFER+
               ADMIT_URGENT+ADMIT_EMERGENCY+IS_NOT_WHITE+IS_MALE, #+log(LOS), 
             data=dat.encounters.final,family="binomial")
 
 model = glm(SEPSIS_BILLING_CODE ~ BASELINE_AGE+factor(ADMIT_SOURCE)+factor(RACE)+
               ADMIT_URGENT+ADMIT_EMERGENCY+IS_MALE+log(LOS),
             data=dat.encounters.final,family="binomial") 
 
 
 auc(roc(model$fitted.values,factor(model$y))) #0.804 #OLD 0.774, before breaking out emergency and urgent
 plot(roc(model$fitted.values,factor(model$y)))
# 
# pred = prediction(model$fitted.values,factor(model$y))
# PR = performance(pred,"prec","rec")
# plot(PR)
# aupr = 0.5*sum( (PR@x.values[[1]][-1]-PR@x.values[[1]][-length(PR@x.values[[1]])])*
#                 (PR@y.values[[1]][-1]+PR@y.values[[1]][-length(PR@y.values[[1]])]), na.rm=T )
# .184 AUPR, not bad...


########## load in labs, write out again with times and a flag for artery vs vein (assume neither=vein)
########## (will log transform some labs and standardize in python; do there for easier conversion later)

lab_files = dir(labs_folder)
for (lab_path in lab_files) {
  dat.lab = fread(paste(labs_folder,lab_path,sep=""))
  lab = strsplit(lab_path,".csv")[[1]]
  dat.lab[, c("PATIENT_MRN","REFERENCE_UNIT") := NULL]
  setkey(dat.lab,ENCOUNTER_ID)
  
  #filter out the IDs that were excluded from the encounters with inner merge
  dat.lab = merge(dat.lab,dat.admits,all.x=F,all.y=F)
  
  #convert time to be relative to admission
  dat.lab[, RESULT_TIME := as.POSIXct(RESULT_TIME, format="%m/%d/%y %H:%M:%S")]
  dat.lab[, TIME := as.numeric(RESULT_TIME - ADMISSION_DATE_TIME)/60.0/60.0]

#   dat.lab[, TIME_FROM_DIS := as.numeric(DISCHARGE_DATE_TIME - RESULT_TIME)/60.0/60.0]
#   
#   print(lab)
#   print(mean(dat.lab$TIME<0))
#   print(median(dat.lab$TIME[dat.lab$TIME<0]))
#   print(mean(dat.lab$TIME_FROM_DIS<0))
#   print(median(dat.lab$TIME_FROM_DIS[dat.lab$TIME_FROM_DIS<0]))  
  
  #for pH, PCO2, Bicarb we need an arterial vs venous indicator
  if (lab=="pH" || lab=="PCO2" || lab=="Bicarbonate") {
    dat.lab[, ARTERIAL := 0]
    dat.lab$ARTERIAL[grep("ARTERIAL",dat.lab$LAB_NAME)] = 1
    dat.lab = dat.lab[,c("ENCOUNTER_ID","TIME","RESULT_VALUE","ARTERIAL"),with=F] #ID
  } else {
    dat.lab = dat.lab[,c("ENCOUNTER_ID","TIME","RESULT_VALUE"),with=F] #ID
  }
  
  #Lactate has some weird values. "", ">20.0", "*" => NA,20,NA
  #use the NAs only to build out feature vectors of labs ordered; no values
  if (lab=="Lactate") { 
    dat.lab$RESULT_VALUE[dat.lab$RESULT_VALUE==">20.0"] = 20.0
    dat.lab[, RESULT_VALUE := as.numeric(RESULT_VALUE)]
  } else {
    dat.lab[, RESULT_VALUE := as.numeric(RESULT_VALUE)]
  }
  
#   par(mfrow=c(1,2))
#   hist(dat.lab$RESULT_VALUE,100,main=lab)
#   hist(log(dat.lab$RESULT_VALUE),100,main=lab)
  
  write.csv(dat.lab,paste("cleaned_data/labs/",lab_path,sep=""),row.names=F,quote=F)
}

####### NOTE: going to want to log-transform many labs with skewed dists.
####### will be done in the modeling file when these are loaded.

#tmp = dat.lab[ENCOUNTER_ID=="137551640",] #Bicarbonate
#tmp2 = dat.encounters.final[ENCOUNTER_ID=="137551640",]

#get some IDs of encounters where labs drawn before admit
# dat.lab.early = dat.lab[TIME < -12,]
# write.csv(dat.lab.early,"early_lactate_labs.csv",row.names=F)




########## create and write out meds; just change the time from date to a time t>=0

med_files = dir(meds_folder)
for (med_path in med_files) {
  if (med_path=="Med_Admin_Times.csv") {next}
  
  dat.med = fread(paste(meds_folder,med_path,sep=""))
  
  #maybe later break down by more specific meds...?
  dat.med[, c("PATIENT_MRN","MED_NAME") := NULL]
  setkey(dat.med,ENCOUNTER_ID)
  
  #filter out the IDs that were excluded from the encounters with inner merge
  dat.med = merge(dat.med,dat.admits,all.x=F,all.y=F)
  
  #convert time to be relative to admission
  dat.med[, MED_START_TIME := as.POSIXct(MED_START_TIME, format="%Y-%m-%d %H:%M:%S")]
  dat.med[, TIME := as.numeric(MED_START_TIME - ADMISSION_DATE_TIME)/60.0/60.0]
  
  #   dat.lab[, TIME_FROM_DIS := as.numeric(DISCHARGE_DATE_TIME - RESULT_TIME)/60.0/60.0]
  # 
    print(med_path)
    print(mean(dat.med$TIME<0,na.rm=T))
    print(median(dat.med$TIME[dat.med$TIME<0],na.rm=T))
  #   print(mean(dat.med$TIME_FROM_DIS<0))
  #   print(median(dat.med$TIME_FROM_DIS[dat.med$TIME_FROM_DIS<0]))
  
  dat.med = dat.med[, c("ENCOUNTER_ID","TIME"),with=F]
  
  write.csv(dat.med,paste("cleaned_data/meds/",med_path,sep=""),row.names=F,quote=F)
}

### this is it for R...?

