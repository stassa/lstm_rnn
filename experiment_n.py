import lstm_rnn
import test_data as datasets
import print_r_plot_files as pr

""" This module is a "test rig" allowing the reproduction of experimental results. 
    It also acts as a handy record of experiments' (latest) settings.
    To reproduce an experiment simply execute the relevant function - for instance, to 
    run Experiment 1 execute the function experiment_1. 

    Function perform_experiment() can be used to plan and execute more experiments. 

    You can also preserve a log of the experiment's command-line output by piping the output
    to a file, for instance with:

    python experiment_n.py > .\experiment_3/experiment_3.log

    Or, to also capture running time statistics on Windows: 

    > Measure-Command { python experiment_n.py > .\experiment_3/experiment_3.log }
    
    Days              : 0
    Hours             : 0
    Minutes           : 11
    Seconds           : 12
    Milliseconds      : 157
    Ticks             : 6721575677
    TotalDays         : 0.00777960147800926
    TotalHours        : 0.186710435472222
    TotalMinutes      : 11.2026261283333
    TotalSeconds      : 672.1575677
    TotalMilliseconds : 672157.5677

    Aye, that was a big one.

    Module test_data.py, imported above, holds the datasets used in all experiments.
    To use a different dataset module, change the name of the import on line 2.
"""

def perform_experiment_n(rfile,training_data,data_size,test_data,binary=[1]
        ,learning_rate=0.1,eras=1,epochs=20,plot_rows=1,plot_cols=1,title='Experiment',sub='',logging=-1,plot_what='er'):
    """ Train the network and output results for one experiment. 
        rfile: a string; the name of an R plot script file to print out. If a path to a sub-directory is given, the sub-directory must exist.
        training_data: a matrix; a sequence of training vectors.
        data_size: the length of a vector in the training data.
        test_data: a matrix; a sequence of target vectors, against which to calculate error of predictions.
        binary: true or false; whether to treat input data as binary and translate to a bit vector for error calculation
        learning_rate: a scalar (float); learning rate for gradient descent.
        eras: a scalar; the number of eras to train for. Each era continues for the given number of epochs.
        epochs: a scalar; the number of epochs to train for.
        plot_rows: a scalar; the number of rows in each sub-plot in the R plot.
        plot_cols: a scalar; the number of columns in each sub-plot in the R plot.
        title: a string; the main title of the R plot.
        sub: a string; the subtitle of the R plot.
    """
    
    # Set the switch controlling whether test data shoudl be translated to bit vectors for error calculation
    lstm_rnn.binary_data = binary

    # Set logging level:
    lstm_rnn.logging_level = logging 
    
    Eras = []

    # Train for the given number of epochs, repeating for the given number of eras.
    for i in range(0,eras):
        one_era = lstm_rnn.unfold_for_n_epochs(training_data, data_size, test_data, learning_rate, epochs)
        Eras.append(one_era['data'])
    
    pr.print_r_plot_file_eras(rfile, Eras, plot_rows, plot_cols, title, sub, plot_what)
    
    cell = one_era['net']
    for i in range(len(test_data)):
        stimulus = test_data[i]
        activation = cell.state_update(stimulus)
        activation = lstm_rnn.reals_to_bit_vector(activation)
        lstm_rnn.logging("Activation for %s: %s" %(stimulus, activation),-1)
    

def experiment_2():
    """ Train a network with a single neuron, no connections from previous layers
        and constant input. The network is trained on a repeating sequence 
        of the binary numerals 0001 and 0010 (1 and 2). The desired outcome is
        to learn that 0010 follows 0001.
    """

    rfile='./experiment_2/experiment_2.r'
    training_data = datasets.binary_counting_constant_train
    data_size = len(datasets.binary_counting_constant_train[0])
    tst_data = datasets.binary_counting_constant_test
    binary=[1]
    learning_rate=0.1
    eras=1
    epochs=40
    plot_rows=1
    plot_cols=1
    title='Experiment 2.'
    sub='Counting in binary (constant input)'

    perform_experiment_n(rfile,training_data,data_size,tst_data,binary
                ,learning_rate,eras,epochs,plot_rows,plot_cols,title,sub)

#experiment_2()


def experiment_3():
    """ Train a network with different learning rate values. 
        NOTE: the R script file generated for this experiment needs a bit of hand-retouching
        to adjust the margins of the main plot figure and to add in the Eta (learning rate) 
        for each Era (those are in the accompanying log file). 
        Make sure to keep a backup if re-generating. 
        If disaster strikes, replace the first line in the file with:
        par(oma=c(1,4,3,1),mfrow=c(2,4))
    """                                                                     

    rfile='./experiment_3/experiment_3.r'
    training_data = datasets.embedded_reber_t
    data_size = len(datasets.embedded_reber_t[0])
    tst_data = datasets.embedded_reber_s
    binary=[1]
    learning_rate=0.01
    eras=8
    epochs=80
    plot_rows=2
    plot_cols=4
    title='Experiment 3.'
    sub='Counting in binary (iterating learning rate)'


    lstm_rnn.binary_data = binary

    lstm_rnn.logging_level = 3 
    
    Eras = []

    for i in range(0,eras):
        print "Epoch learning rate: %s ==============================" % learning_rate
        one_era = lstm_rnn.unfold_for_n_epochs(training_data, data_size, tst_data, learning_rate, epochs)
        Eras.append(one_era['data'])
        learning_rate = learning_rate +.10 
    
    pr.print_r_plot_file_eras(rfile, Eras, plot_rows, plot_cols, title, sub, 'er')
    
    cell = one_era['net']
    for i in range(len(tst_data)):
        stimulus = tst_data[i]
        activation = cell.state_update(stimulus)
        activation = lstm_rnn.reals_to_bit_vector(activation)
        lstm_rnn.logging("Activation for %s: %s" %(stimulus, activation),-1)

#experiment_3()


def experiment_5():
    """ Train a deep network with two LSTM cells stacked one on top of the other.
    """                                                                     

    rfile='./experiment_5/experiment_5.r'
    training_data = datasets.binary_counting_train
    data_size = len(datasets.binary_counting_train[0])
    tst_data = datasets.binary_counting_test
    binary=[1]
    learning_rate=0.05
    eras=10
    epochs=80
    plot_rows=2
    plot_cols=5 
    title='Experiment 5.'
    sub='Counting in binary (stacked cells)'


    lstm_rnn.binary_data = binary

    lstm_rnn.logging_level = 3 
    
    Eras = []

    for i in range(0,eras):
        one_era = lstm_rnn.deep_unfold_for_n_epochs(training_data, data_size, tst_data, learning_rate, epochs)
        Eras.append(one_era['data'])
        learning_rate = learning_rate +.10 
    
    pr.print_r_plot_file_eras(rfile, Eras, plot_rows, plot_cols, title, sub, 'er')
    
    cell1, cell2 = one_era['block']
    for i in range(len(tst_data)):
        stimulus = tst_data[i]
        # Activate the first layer
        lstm_rnn.logging("cell 1 activation ======================",2)
        activation = cell1.state_update(stimulus)
        # Propagate activation forwards
        cell2.ht_min_1 = activation
        # Activate the second layer
        lstm_rnn.logging("cell 2 activation ======================",2)
        activation = cell2.state_update(stimulus)
        activation = lstm_rnn.reals_to_bit_vector(activation)
        lstm_rnn.logging("Activation for %s: %s" %(stimulus, activation),-1)

#experiment_5()


def experiment_6():
    """ Train the network on a sequence of binary numerals, from 0 to 9 (bin). 
        The desired outcome is for the network to learn that, for example, 
        [0,1,0,1] is followed by [0,1,1,0]
    """                                                                     

    rfile='./experiment_6/experiment_6_8_eras.r'
    training_data = datasets.binary_counting_train
    data_size = len(datasets.binary_counting_train[0])
    tst_data = datasets.binary_counting_test
    binary=[1]
    learning_rate=0.1
    eras=8
    epochs=80
    plot_rows=2
    plot_cols=4
    title='Experiment 6.'
    sub='Counting in binary'

    perform_experiment_n(rfile,training_data,data_size,tst_data,binary
                ,learning_rate,eras,epochs,plot_rows,plot_cols,title,sub)

#experiment_6()    


def experiment_7():
    """ 
    """                                                                     

    rfile='./experiment_7/experiment_7.r'
    training_data = datasets.reber_t
    data_size = len(datasets.reber_t[0])
    tst_data = datasets.reber_s
    binary=[1]
    learning_rate=0.1
    eras=8
    epochs=80
    plot_rows=2
    plot_cols=4
    title='Experiment 7.'
    sub='Learning the Reber grammar'

    perform_experiment_n(rfile,training_data,data_size,tst_data,binary
                ,learning_rate,eras,epochs,plot_rows,plot_cols,title,sub)

#experiment_7()   


def experiment_8():
    """ Train a network on the embedded Reber grammar.
        The target is still the next element in the sequence.
    """                                                                     

    rfile='./experiment_8/experiment_8.r'
    training_data = datasets.embedded_reber_t
    data_size = len(datasets.embedded_reber_t[0])
    tst_data = datasets.embedded_reber_s
    binary=[1]
    learning_rate=0.1
    eras=4
    epochs=80
    plot_rows=2
    plot_cols=4
    title='Experiment 8.'
    sub='Learning the Embedded Reber grammar'

    perform_experiment_n(rfile,training_data,data_size,tst_data,binary
                ,learning_rate,eras,epochs,plot_rows,plot_cols,title,sub)

    rfile_ea='./experiment_8/experiment_8_ea.r'

    perform_experiment_n(rfile_ea,training_data,data_size,tst_data,binary
                ,learning_rate,eras,epochs,plot_rows,plot_cols,title,sub,-1,'ea')


experiment_8()
