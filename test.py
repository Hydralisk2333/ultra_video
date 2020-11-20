import glob

dirPath = 'F:\\dataset\\dual_data\\0\\ultra\\*.wav'
for path in glob.glob(dirPath):
    print(path)