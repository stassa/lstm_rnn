par(oma=c(1,4,3,1),mfrow=c(1,1))
e0 <- c(4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5)

plot(e0,col='blue',type='l',xlab='Epoch',ylab='Error',main='Era 0')

title(main='Experiment 2.',outer=TRUE)
mtext(side=1, 'Counting in binary (constant input)', outer=TRUE)
