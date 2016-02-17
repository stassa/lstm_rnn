import numpy as np

def print_r_plot_file_eras(outfile,eras,sub_plot_rows,sub_plot_cols,plot_title='',plot_sub_title='',plot_what='er'):
    """ Print an R plot file for all Eras of an experiment. 
        Each Era may have any number of Epochs 
        in each of which we present the network with the entire input once.
    """

    # Errors of all epochs
    errors = []
    # Activations of all epochs
    activations = []

    for era in eras:
        e,a = era
        errors += [e]
        activations += [a]

    # Plot errors over epochs.
    if plot_what == 'er':

        with open(outfile, "w") as of:

            # Partition our R graph. Also adjust margins.
            of.write("par(oma=c(1,4,3,1),mfrow=c(%s,%s))\n" %(sub_plot_rows, sub_plot_cols))

            # Note well: a single epoch will cause an error in R like this one: 
            # "Error in c(3.5, ) : argument 2 is empty"
            # c(3.5,) is a concatenation of 3.5 and another element- but there is no other element
            # because we only ran for a single epoch. 
            # Just remove the comma from the vector by hand :0
            for error in enumerate(errors):
                i,e = error
                of.write("e%s <- c%s\n" %(i, tuple(e),))

            of.write("\n") 
            for error in enumerate(errors):
                of.write("plot(e%s,col='blue',type='l',xlab='Epoch',ylab='Error',main='Era %s')\n" %(error[0],error[0]))

            of.write("\n") 
            of.write("title(main='%s',outer=TRUE)\n" %(plot_title))
            of.write("mtext(side=1, '%s', outer=TRUE)\n" %(plot_sub_title))

    # Plot errors and activations together.
    # Try with: eras = 4, sub_rows = 2, sub_cols = 4
    if plot_what == 'ea':

        with open(outfile, "w") as of:

            # Partition our R graph. 
            of.write("par(oma=c(1,4,3,1),mfrow=c(%s,%s))\n" %(sub_plot_rows, sub_plot_cols))
            
            for error in enumerate(errors):
                i,e = error
                of.write("e%s <- c%s\n" %(i, tuple(e),))

            for activation in enumerate(activations):
                i,a = activation
                mean_activations = [np.mean(actv) for actv in a] 
                of.write("a%s <- c%s\n" %(i, tuple(mean_activations),))

            of.write("\n") 
            for error in enumerate(errors):
                of.write("plot(e%s,col='blue',type='l',xlab='Epoch',ylab='Error',main='Era %s')\n" %(error[0],error[0]))

            of.write("\n")
            for activation in enumerate(activations):
                of.write("plot(a%s,col='red',type='l', xlab='Epoch',ylab='Mean Activation',main='Era %s')\n" %(activation[0],activation[0]))

            of.write("\n") 
            of.write("title(main='%s',outer=TRUE)\n" %(plot_title))
            of.write("mtext(side=1, '%s', outer=TRUE)\n" %(plot_sub_title))



def print_r_plot_file(outfile,activations,plot_what='ty'):
    """ 
        Deprecated- vestiges of an earlier version. use print_r_plot_file_eras() instead

        Generate R file to create plots. Could also do in matplotlib (but am more familiar with R plots)
        
        outfile: name of the r file to print out.
        activations: tuple of block layer activations, plus input vector.
        plot_what: string in {'ty',xy}. 'ty' plots activations against time steps. 'xy' plots activations
                    against their respective inputs.

        All this won't quite work with len(v[0]) > 1.
        But, can always try... 
    """

    x,i,m,f,r,ct,y,H = activations

    with open(outfile, "w") as of:

        of.write("x <- c%s\n"  %(tuple([xt[0] for xt in x]),))
        of.write("i <- c%s\n"  %(tuple(i),))
        of.write("m <- c%s\n"  %(tuple(m),))
        of.write("f <- c%s\n"  %(tuple(f),))
        of.write("r <- c%s\n"  %(tuple(r),))
        of.write("ct <- c%s\n" %(tuple(ct),))
        of.write("y <- c%s\n" %(tuple(y),))
        of.write("H <- %s\n" % H)
        
        of.write("par(mfrow=c(2,3))\n")

        # Plot inputs against activations.
        if plot_what == 'xy':
        
            of.write("plot(x,y,col='blue',type='l', xlab='cell input', ylab='cell activation')\n")
            of.write("plot(i,y,col='red',type='l', xlab='input layer', ylab='cell activation')\n")
            of.write("plot(m,y,col='darkgreen',type='l', xlab='remember gate', ylab='cell activation')\n")
            of.write("plot(f,y,col='darkmagenta',type='l', xlab='forget gate', ylab='cell activation')\n")
            of.write("plot(r,y,col='darkred',type='l', xlab='recall gate', ylab='cell activation')\n")
            of.write("plot(ct,y,col='cornflowerblue',type='l', xlab='cell memory', ylab='cell activation')\n")

        # Plot activations against time.
        elif plot_what == 'ty':        
        
            of.write("plot(y,col='blue',type='l', ylab='time step', xlab='cell activation')\n")
            of.write("plot(i,col='red',type='l', ylab='time step', xlab='input layer')\n")
            of.write("plot(m,col='darkgreen',type='l', ylab='time step', xlab='remember gate')\n")
            of.write("plot(f,col='darkmagenta',type='l', ylab='time step', xlab='forget gate')\n")
            of.write("plot(r,col='darkred',type='l', ylab='time step', xlab='recall gate')\n")
            of.write("plot(ct,col='cornflowerblue',type='l', ylab='time step', xlab='cell memory')\n")


def print_r_plot_file_stacked(outfile,activations, plot_what = 'ty'):
    """ 
        Deprecated- vestiges of an earlier version. use print_r_plot_file_eras() instead

        Generate R plot file. 
        
        outfile: name of the r file to print out.
        activations: tuple of block layer activation tuples, plus input vectors.

        NOTE: producing source code like this tends to get a bit repetitive. This is
              even more so the case here, because we're plotting two sets of activations
              both with the same structure.
    """

    # Unpack activation tuples. 
    block_1_activations, block_2_activations = activations
    x1,i1,m1,f1,r1,ct1,y1 = block_1_activations
    x2,i2,m2,f2,r2,ct2,y2 = block_2_activations

    with open(outfile, "w") as of:

        # The input to block 1 is a matrix of 1-d vectors
        of.write("x1 <- c%s\n"  %(tuple([xt[0] for xt in x1]),))
        # The input to block 2 is a vector- the vector of outputs of block 1
        of.write("x2 <- c%s\n"  %(tuple(x2),))

        of.write("i1 <- c%s\n"  %(tuple(i1),))
        of.write("i2 <- c%s\n"  %(tuple(i2),))
        
        of.write("m1 <- c%s\n"  %(tuple(m1),))
        of.write("m2 <- c%s\n"  %(tuple(m2),))
        
        of.write("f1 <- c%s\n"  %(tuple(f1),))
        of.write("f2 <- c%s\n"  %(tuple(f2),))
        
        of.write("r1 <- c%s\n"  %(tuple(r1),))
        of.write("r2 <- c%s\n"  %(tuple(r2),))
        
        of.write("ct1 <- c%s\n" %(tuple(ct1),))
        of.write("ct2 <- c%s\n" %(tuple(ct2),))
        
        of.write("y1 <- c%s\n" %(tuple(y1),))
        of.write("y2 <- c%s\n" %(tuple(y2),))
        
        of.write("par(mfrow=c(3,4))\n")

        # Plot input vectors against activation vectors.
        if plot_what == 'xy':

            # Each vector will be plotted in its own sub-plot. Scales differe so it's easier this way
            of.write("plot(x1,y1,col='blue',type='l', xlab='cell 1 input', ylab='cell 1 activation')\n")
            of.write("plot(x2,y2,col='darkblue',type='l', xlab='cell 2 input', ylab='cell 2 activation')\n")
            
            of.write("plot(i1,y1,col='red',type='l', xlab='cell 1 input layer', ylab='cell 1 activation')\n")
            of.write("plot(i2,y2,col='darkred',type='l', xlab='cell 2 input layer', ylab='cell 2 activation')\n")
            
            of.write("plot(m1,y1,col='green',type='l', xlab='cell 1 remember gate', ylab='cell 1 activation')\n")
            of.write("plot(m2,y2,col='darkgreen',type='l', xlab='cell 2 remember gate', ylab='cell 2 activation')\n")
            
            of.write("plot(f1,y1,col='magenta',type='l', xlab='cell 1 forget gate', ylab='cell 1 activation')\n")
            of.write("plot(f2,y2,col='darkmagenta',type='l', xlab='cell 2 forget gate', ylab='cell 2 activation')\n")
            
            of.write("plot(r1,y1,col='chocolate4',type='l', xlab='cell 1 recall gate', ylab='cell 1 activation')\n")
            of.write("plot(r2,y2,col='coral4',type='l', xlab='cell 2 recall gate', ylab='cell 2 activation')\n")
            
            of.write("plot(ct1,y1,col='burlywood',type='l', xlab='cell 1 memory', ylab='cell 1 activation')\n")
            of.write("plot(ct2,y2,col='burlywood4',type='l', xlab='cell 2 memory', ylab='cell 2 activation')\n")

       
        # Plot activations against time step. 
        elif plot_what == 'ty': 

            of.write("plot(y1,col='blue',type='l', xlab='time step', ylab='cell 1 activation')\n")
            of.write("plot(y2,col='darkblue',type='l', xlab='time step', ylab='cell 2 activation')\n")
            
            of.write("plot(i1,col='red',type='l', xlab='time step', ylab='cell 1 input layer')\n")
            of.write("plot(i2,col='darkred',type='l', xlab='time step', ylab='cell 2 input layer')\n")
            
            of.write("plot(m1,col='green',type='l', xlab='time step', ylab='cell 1 remember gate')\n")
            of.write("plot(m2,col='darkgreen',type='l', xlab='time step', ylab='cell 2 remember gate')\n")
            
            of.write("plot(f1,col='magenta',type='l', xlab='time step', ylab='cell 1 forget gate')\n")
            of.write("plot(f2,col='darkmagenta',type='l', xlab='time step', ylab='cell 2 forget gate')\n")
            
            of.write("plot(r1,col='chocolate4',type='l', xlab='time step', ylab='cell 1 recall gate')\n")
            of.write("plot(r2,col='coral4',type='l', xlab='time step', ylab='cell 2 recall gate')\n")
            
            of.write("plot(ct1,col='burlywood',type='l', xlab='time step', ylab='cell 1 memory')\n")
            of.write("plot(ct2,col='burlywood4',type='l', xlab='time step', ylab='cell 2 memory')\n")
        
        else:
            print "Nothing to plot. Goodbye."

