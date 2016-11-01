library(data.table)

##Load Data##

#Encounter
enc = fread("/Users/Sanjay/Desktop/Research/DIHI/Data/Encounter/Sepsis_EncounterDetails_Outcomes.csv")

#Demographics
dem = fread("/Users/Sanjay/Desktop/Research/DIHI/Data/Demographics/sepsis_pt_demographics.csv")

#Medications
med = fread("/Users/Sanjay/Desktop/Research/DIHI/Data/Medications/Med_Admin_Times.csv")

#Labs
pre_dir = "/Users/Sanjay/Desktop/Research/DIHI/Data/Labs/"
files = dir(pre_dir)
Labs = data.frame()
for (file in files){
  dat = fread(paste0(pre_dir, file))
  dat$LAB_NAME = dat$LAB_NAME[1] #Make each lab file have one unique lab name
  Labs = rbind(Labs, dat)
}
