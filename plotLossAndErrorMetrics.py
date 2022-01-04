import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def plotLossAndErrorMetrics(filename):
    fileOutDf = pd.read_csv(filename, sep = '\t', header=None)
    matplotlib.rc('font', size=10)
    
    plt.subplot(2,1,1)
    plt.plot(fileOutDf[1], fileOutDf[2])
    plt.plot(fileOutDf[1], fileOutDf[3])
    plt.yscale("log")
    plt.legend(['Training','Validation'])
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.grid()
    

    plt.subplot(2,1,2)
    plt.plot(fileOutDf[1], fileOutDf[4])
    plt.plot(fileOutDf[1], fileOutDf[5])
    plt.plot(fileOutDf[1], fileOutDf[6])
    plt.legend(['WER','BLEU','GLEU'])
    plt.xlabel('Batch number')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

    print(max(fileOutDf[4]))




filename = 'D:/OneDrive/Skrivebord/adam3rdTraining.txt'

plotLossAndErrorMetrics(filename)