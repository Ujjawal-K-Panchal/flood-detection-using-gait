
tr<-read.csv("trsens.csv")
tst<-read.csv("tstsens.csv")

myfit = rpart(lbl~.,method="class", data=tr,
control=rpart.control(minsplit=50), parms=list(split='information'))
tstres = predict(myfit, tst, type="class")
print("Classification Accuracy:")
print(length(tstres[tstres==tst$lbl])/length(tst$lbl))
print("Confusion Matrix (Actual vs Predicted):")
table(tst$lbl, tstres)


