#/usr/bin/env/python3
import math

#################################################################################
## Created by Ying Shi Teh
## Last modified on July 5, 2021
#################################################################################

##-------------------------------------------------------------------------------
## Combine Python files
##-------------------------------------------------------------------------------
def combinePythonFiles(YdataType,nfft,dataSize,batchSize,totalBatches):
    dataY = np.empty((dataSize,nfft))
    for batchNumber in range(1,totalBatches+1):
        minDataNumber = batchSize*(batchNumber-1)
        maxDataNumber = min(batchSize*batchNumber, dataSize)
        filename = 'mg_Y_' + YdataType + '_batch' + str(batchNumber) + '.npy'
        dataY[minDataNumber:maxDataNumber] = np.load(filename)

    filename = 'mg_Y_' + YdataType + '.npy'
    np.save(filename,dataY)

    return

##-------------------------------------------------------------------------------
## Combine text files
##-------------------------------------------------------------------------------
def combineTextFiles(YdataType,totalBatches):
    # Filenames 
    filenames = []
    for batchNumber in range(1,totalBatches+1):
        filenames.append('check_total_' + YdataType + '_batch' + str(batchNumber) + '.txt')
    with open('check_total' + YdataType + '.txt') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():
    dataSize = 3000
    batchSize = 20
    totalBatches = int(math.ceil(float(dataSize)/float(batchSize)))
    ngfft = [36,64,60]
    nfft = ngfft[0]*ngfft[1]*ngfft[2]

    # Combine python files
    combinePythonFiles('EBAND',nfft,dataSize,batchSize,totalBatches)
    combinePythonFiles('ENTR',nfft,dataSize,batchSize,totalBatches)

    # Combine text files
    combineTextFiles('band_energy',totalBatches)
    combineTextFiles('entropy',totalBatches)

if __name__ == '__main__': main()
