rm(list = setdiff(ls(), lsf.str()))
mydata=read.csv("merged_windowv2.csv")

# First process data, and create training and test files

library(rpart)
nonfl<-mydata[mydata$lbl == 0,]
fl1<-mydata[mydata$lbl == 0.19,]
fl2<-mydata[mydata$lbl == 2.5,]
fl3<-mydata[mydata$lbl == 4.5,]

totnonfl=length(nonfl$lbl)
totfl1=length(fl1$lbl)
totfl2=length(fl2$lbl)
totfl3=length(fl3$lbl)
ntrnonfl=as.integer(totnonfl*0.8)
ntrfl1=as.integer(totfl1*0.8)
ntrfl2=as.integer(totfl2*0.8)
ntrfl3=as.integer(totfl3*0.8)
trnonfl<-nonfl[1:ntrnonfl,]
trfl1<-fl1[1:ntrfl1,]
trfl2<-fl2[1:ntrfl2,]
trfl3<-fl3[1:ntrfl3,]
tstnonfl<-nonfl[(ntrnonfl+1):totnonfl,]
tstfl1<-fl1[(ntrfl1+1):totfl1,]
tstfl2<-fl2[(ntrfl2+1):totfl2,]
tstfl3<-fl3[(ntrfl3+1):totfl3,]
tr <- trnonfl
tr <- rbind(tr, trfl1)
tr <- rbind(tr, trfl2)
tr <- rbind(tr, trfl3)
tst <-tstnonfl
tst <-rbind(tst,tstfl1)
tst <-rbind(tst,tstfl2)
tst <-rbind(tst,tstfl3)

write.csv(tr,"trsens.csv", row.names=FALSE)
write.csv(tst,"tstsens.csv", row.names=FALSE)

