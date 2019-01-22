

library(rpart.plot)
setEPS()
postscript("dtree.eps")
rpart.plot(myfit, type=4, extra=2, clip.right.labs=FALSE,
varlen=0, faclen=0)
dev.off()

